"""
Multi-Format Document Loader for Advanced RAG System
Handles loading various document types including PDFs, Word docs, Excel files, and text files
"""

import tempfile
import os
from typing import List
from pathlib import Path
import logging

from langchain_core.documents import Document
from backend.multimodal_loader import MultiFormatDocumentLoader as BaseMultiFormatLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalDocumentLoader:
    """Multi-format document loader for API usage"""

    def __init__(self):
        """Initialize with the base multi-format loader"""
        self.base_loader = BaseMultiFormatLoader()

    def load_document(self, file_path: str) -> List[Document]:
        """Load a document from file path using the multi-format loader"""
        return self.base_loader.load_document(file_path)

    def load_file_content(self, file_content: bytes, filename: str) -> List[Document]:
        """
        Loads a document from file content bytes

        Args:
            file_content: Raw file content as bytes
            filename: Original filename with extension

        Returns:
            List[Document]: Document chunks from the file content
        """
        # Get file extension
        file_extension = filename.split(".")[-1].lower()

        # Check if file type is supported
        if not self.base_loader.is_supported_format(f"dummy.{file_extension}"):
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Create temporary file to work with loaders that need file paths
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{file_extension}",
            prefix=f"uploaded_{filename.split('.')[0]}_",
        ) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            logger.info(
                f"Processing file: {filename} (size: {len(file_content)} bytes)"
            )

            # Load document using the loader
            documents = self.base_loader.load_document(tmp_file_path)

            # Update metadata with original filename and upload info
            for doc in documents:
                doc.metadata.update(
                    {
                        "original_filename": filename,
                        "upload_size": len(file_content),
                        "processed_via": "api_upload",
                    }
                )

            logger.info(
                f"Successfully processed {filename}: {len(documents)} chunks extracted"
            )
            return documents

        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise Exception(f"Failed to process file {filename}: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except OSError:
                logger.warning(f"Could not delete temporary file: {tmp_file_path}")

    def load_multiple_files(self, file_data_list: List[tuple]) -> List[Document]:
        """
        Loads multiple documents from file content

        Args:
            file_data_list: List of tuples (file_content_bytes, filename)

        Returns:
            List[Document]: Combined document chunks from all files
        """
        all_documents = []
        failed_files = []

        for file_content, filename in file_data_list:
            try:
                documents = self.load_file_content(file_content, filename)
                all_documents.extend(documents)
                logger.info(f"Successfully loaded {filename}")
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {str(e)}")
                failed_files.append(filename)

        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")

        logger.info(
            f"Total: {len(all_documents)} document chunks from {len(file_data_list) - len(failed_files)} successful uploads"
        )
        return all_documents

    def get_supported_extensions(self) -> List[str]:
        """Returns a list of supported file extensions"""
        return self.base_loader.get_supported_extensions()

    def get_supported_extensions_display(self) -> str:
        """Returns a formatted string of supported extensions for display"""
        extensions = self.get_supported_extensions()
        return ", ".join([f".{ext}" for ext in sorted(extensions)])

    def is_supported_file(self, filename: str) -> bool:
        """Checks if a filename has a supported extension"""
        return self.base_loader.is_supported_format(filename)

    def get_file_info(self, filename: str, file_size: int) -> dict:
        """
        Gets information about a file

        Args:
            filename: The filename
            file_size: Size of the file in bytes

        Returns:
            dict: File information including name, size, and type
        """
        file_extension = filename.split(".")[-1].lower()

        return {
            "filename": filename,
            "size": file_size,
            "extension": file_extension,
            "is_supported": self.is_supported_file(filename),
        }
