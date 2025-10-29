"""
Hybrid Metadata Extraction Layer for Advanced RAG System

This module implements a two-tier metadata extraction strategy:
1. Fast regex-based extraction for structured patterns (dates, emails, IDs, etc.)
2. Lightweight LLM-based extraction for semantic metadata (titles, authors, summaries)

Design Goals:
- Minimal latency overhead (<1 second per document)
- Format-agnostic extraction (works with PDF, DOCX, TXT, XLSX, etc.)
- Scalable and efficient (parallel processing, caching)
- Preserves original metadata while enriching with content metadata
"""

import re
import logging
import hashlib
from typing import List, Dict, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Hybrid metadata extraction system combining regex-based and LLM-based extraction
    """
    
    # Regex patterns for fast structured metadata detection
    PATTERNS = {
        # Date patterns (various formats)
        'dates': [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{2}/\d{2}/\d{4}\b',  # DD/MM/YYYY or MM/DD/YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
        ],
        
        # Email addresses
        'emails': [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        ],
        
        # Phone numbers (various formats)
        'phone_numbers': [
            r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US/International
            r'\b\d{3}-\d{3}-\d{4}\b',  # XXX-XXX-XXXX
        ],
        
        # Document/Invoice numbers
        'document_ids': [
            r'\b(?:INV|INVOICE)[-\s]?#?\d{4,}\b',  # Invoice numbers
            r'\b(?:DOC|DOCUMENT)[-\s]?#?\d{4,}\b',  # Document numbers
            r'\b(?:PO|ORDER)[-\s]?#?\d{4,}\b',  # Purchase orders
            r'\b(?:REF|REFERENCE)[-\s]?#?\d{4,}\b',  # Reference numbers
        ],
        
        # URLs and DOIs
        'urls': [
            r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
        ],
        
        'dois': [
            r'\b10\.\d{4,}/[-._;()/:A-Za-z0-9]+\b',
        ],
        
        # Currency amounts
        'currency': [
            r'\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # USD
            r'€\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # EUR
            r'£\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # GBP
        ],
    }
    
    # Metadata cue patterns for identifying metadata-rich regions
    METADATA_CUES = [
        r'(?i)(?:title|subject|topic):\s*(.+)',
        r'(?i)(?:author|by|written by|prepared by):\s*(.+)',
        r'(?i)(?:date|dated|published):\s*(.+)',
        r'(?i)(?:organization|company|institution):\s*(.+)',
        r'(?i)(?:department|division|team):\s*(.+)',
        r'(?i)(?:version|revision|edition):\s*(.+)',
        r'(?i)(?:status|classification):\s*(.+)',
    ]
    
    def __init__(self, use_llm: bool = True, cache_enabled: bool = True):
        """
        Initialize the metadata extractor
        
        Args:
            use_llm: Whether to use LLM for semantic metadata extraction
            cache_enabled: Whether to cache extraction results
        """
        self.use_llm = use_llm
        self.cache_enabled = cache_enabled
        self.extraction_cache = {}
        
        # Initialize LLM for semantic extraction (Gemini Flash for speed)
        if self.use_llm:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0,
                    max_tokens=500,  # Keep responses concise
                )
                logger.info("✅ Initialized Gemini Flash for metadata extraction")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize LLM: {e}. Falling back to regex-only mode.")
                self.use_llm = False
        
        # Create LLM extraction prompt
        self._create_llm_prompt()
    
    def _create_llm_prompt(self):
        """Create optimized prompt for LLM-based metadata extraction"""
        prompt_text = """You are a metadata extraction specialist. Extract key metadata from the document snippet below.

DOCUMENT SNIPPET (first ~500 characters):
{text_snippet}

Extract the following metadata if present:
1. **Title**: Main document title or subject
2. **Author(s)**: Person(s) who created the document
3. **Organization**: Company, institution, or organization
4. **Date**: Any date mentioned (publication, creation, etc.)
5. **Document Type**: Type of document (report, invoice, paper, resume, etc.)
6. **Summary**: One-sentence summary of the document's purpose

Respond ONLY with a JSON object (no markdown, no explanation):
{{
  "title": "extracted title or null",
  "authors": ["author1", "author2"] or [],
  "organization": "organization name or null",
  "date": "date string or null",
  "document_type": "type or null",
  "summary": "one-sentence summary or null"
}}"""
        
        self.llm_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise metadata extraction assistant. Always respond with valid JSON only."),
            ("user", prompt_text)
        ])
    
    def _get_document_hash(self, text: str) -> str:
        """Generate hash for caching"""
        return hashlib.md5(text[:1000].encode()).hexdigest()
    
    def _extract_front_matter(self, text: str, max_chars: int = 1000) -> str:
        """
        Extract the front matter of a document (likely to contain metadata)
        
        Args:
            text: Full document text
            max_chars: Maximum characters to extract
            
        Returns:
            Front matter text snippet
        """
        # Take first N characters
        front_matter = text[:max_chars]
        
        # Try to break at a natural boundary (paragraph, sentence)
        if len(text) > max_chars:
            # Find last period or newline within the limit
            last_period = front_matter.rfind('.')
            last_newline = front_matter.rfind('\n\n')
            
            boundary = max(last_period, last_newline)
            if boundary > max_chars // 2:  # Only use if it's not too short
                front_matter = front_matter[:boundary + 1]
        
        return front_matter.strip()
    
    def _extract_regex_metadata(self, text: str) -> Dict[str, Any]:
        """
        Fast regex-based extraction of structured metadata
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        
        # Extract each pattern type
        for pattern_type, patterns in self.PATTERNS.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                # Deduplicate and limit results
                unique_matches = list(set(matches))[:5]  # Max 5 per type
                metadata[pattern_type] = unique_matches
        
        # Extract metadata cues (title, author, etc.)
        cue_metadata = {}
        for cue_pattern in self.METADATA_CUES:
            match = re.search(cue_pattern, text[:2000], re.IGNORECASE | re.MULTILINE)
            if match:
                key = match.group(0).split(':')[0].strip().lower()
                value = match.group(1).strip()
                cue_metadata[key] = value
        
        if cue_metadata:
            metadata['cue_based'] = cue_metadata
        
        return metadata
    
    def _extract_llm_metadata(self, text: str) -> Dict[str, Any]:
        """
        LLM-based extraction of semantic metadata
        
        Args:
            text: Document text (will use front matter only)
            
        Returns:
            Dictionary of extracted semantic metadata
        """
        if not self.use_llm:
            return {}
        
        try:
            # Extract front matter for LLM analysis (keep it small for speed)
            front_matter = self._extract_front_matter(text, max_chars=500)
            
            # Check cache
            if self.cache_enabled:
                cache_key = self._get_document_hash(front_matter)
                if cache_key in self.extraction_cache:
                    logger.debug("🚀 Cache hit for LLM metadata extraction")
                    return self.extraction_cache[cache_key]
            
            # Create chain and invoke
            chain = self.llm_prompt | self.llm
            response = chain.invoke({"text_snippet": front_matter})
            
            # Parse JSON response
            import json
            response_text = response.content.strip()
            
            # Remove markdown code fences if present
            if response_text.startswith('```'):
                response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)
            
            llm_metadata = json.loads(response_text)
            
            # Cache result
            if self.cache_enabled:
                self.extraction_cache[cache_key] = llm_metadata
            
            return llm_metadata
            
        except Exception as e:
            logger.warning(f"⚠️ LLM metadata extraction failed: {e}")
            return {}
    
    def extract_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Extract metadata from documents using hybrid approach
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Document objects with enriched metadata
        """
        start_time = datetime.now()
        logger.info(f"🔍 Starting metadata extraction for {len(documents)} documents")
        
        enriched_documents = []
        
        for idx, doc in enumerate(documents):
            try:
                # Get document text
                text = doc.page_content
                
                # Skip if document is too short
                if len(text.strip()) < 50:
                    logger.debug(f"Skipping metadata extraction for very short document {idx}")
                    enriched_documents.append(doc)
                    continue
                
                # Phase 1: Fast regex-based extraction (runs on all documents)
                regex_metadata = self._extract_regex_metadata(text)
                
                # Phase 2: LLM-based extraction (only on front matter)
                llm_metadata = {}
                if self.use_llm and idx == 0:  # Only extract from first document chunk
                    llm_metadata = self._extract_llm_metadata(text)
                
                # Combine metadata
                content_metadata = {
                    'structured': regex_metadata,  # Regex-extracted patterns
                    'semantic': llm_metadata,      # LLM-extracted metadata
                    'extraction_timestamp': datetime.now().isoformat(),
                }
                
                # Preserve original metadata and add content_metadata
                enriched_metadata = doc.metadata.copy() if doc.metadata else {}
                enriched_metadata['content_metadata'] = content_metadata
                
                # Create new document with enriched metadata
                enriched_doc = Document(
                    page_content=doc.page_content,
                    metadata=enriched_metadata
                )
                
                enriched_documents.append(enriched_doc)
                
            except Exception as e:
                logger.error(f"❌ Error extracting metadata from document {idx}: {e}")
                # On error, keep original document
                enriched_documents.append(doc)
        
        # Calculate latency
        end_time = datetime.now()
        latency = (end_time - start_time).total_seconds()
        
        logger.info("✅ Metadata extraction complete:")
        logger.info(f"   Documents processed: {len(documents)}")
        logger.info(f"   Total latency: {latency:.3f}s")
        logger.info(f"   Avg latency per doc: {latency/len(documents):.3f}s")
        
        if latency > 1.0:
            logger.warning(f"⚠️ Latency exceeded 1 second target: {latency:.3f}s")
        
        return enriched_documents
    
    def extract_metadata_parallel(self, documents: List[Document], max_workers: int = 4) -> List[Document]:
        """
        Extract metadata from documents using parallel processing
        
        Args:
            documents: List of Document objects
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of Document objects with enriched metadata
        """
        start_time = datetime.now()
        logger.info(f"🔍 Starting parallel metadata extraction for {len(documents)} documents")
        
        def process_single_document(doc_tuple: Tuple[int, Document]) -> Tuple[int, Document]:
            """Process a single document and return (index, enriched_doc)"""
            idx, doc = doc_tuple
            
            try:
                text = doc.page_content
                
                # Skip if document is too short
                if len(text.strip()) < 50:
                    return (idx, doc)
                
                # Phase 1: Fast regex-based extraction
                regex_metadata = self._extract_regex_metadata(text)
                
                # Phase 2: LLM-based extraction (only on first document chunk)
                llm_metadata = {}
                if self.use_llm and idx == 0:
                    llm_metadata = self._extract_llm_metadata(text)
                
                # Combine metadata
                content_metadata = {
                    'structured': regex_metadata,
                    'semantic': llm_metadata,
                    'extraction_timestamp': datetime.now().isoformat(),
                }
                
                # Preserve original metadata and add content_metadata
                enriched_metadata = doc.metadata.copy() if doc.metadata else {}
                enriched_metadata['content_metadata'] = content_metadata
                
                # Create new document with enriched metadata
                enriched_doc = Document(
                    page_content=doc.page_content,
                    metadata=enriched_metadata
                )
                
                return (idx, enriched_doc)
                
            except Exception as e:
                logger.error(f"❌ Error extracting metadata from document {idx}: {e}")
                return (idx, doc)
        
        # Process documents in parallel
        enriched_documents = [None] * len(documents)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_doc = {
                executor.submit(process_single_document, (idx, doc)): idx
                for idx, doc in enumerate(documents)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_doc):
                try:
                    idx, enriched_doc = future.result()
                    enriched_documents[idx] = enriched_doc
                except Exception as e:
                    logger.error(f"❌ Parallel processing error: {e}")
        
        # Calculate latency
        end_time = datetime.now()
        latency = (end_time - start_time).total_seconds()
        
        logger.info("✅ Parallel metadata extraction complete:")
        logger.info(f"   Documents processed: {len(documents)}")
        logger.info(f"   Total latency: {latency:.3f}s")
        logger.info(f"   Avg latency per doc: {latency/len(documents):.3f}s")
        logger.info(f"   Workers used: {max_workers}")
        
        if latency > 1.0:
            logger.warning(f"⚠️ Latency exceeded 1 second target: {latency:.3f}s")
        
        return enriched_documents
    
    def clear_cache(self):
        """Clear the extraction cache"""
        self.extraction_cache.clear()
        logger.info("🗑️ Metadata extraction cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.extraction_cache),
            'cache_enabled': self.cache_enabled,
        }


# Convenience function for quick metadata extraction
def extract_metadata(documents: List[Document], use_llm: bool = True, parallel: bool = False) -> List[Document]:
    """
    Convenience function to extract metadata from documents
    
    Args:
        documents: List of Document objects
        use_llm: Whether to use LLM for semantic extraction
        parallel: Whether to use parallel processing
        
    Returns:
        List of Document objects with enriched metadata
    """
    extractor = MetadataExtractor(use_llm=use_llm)
    
    if parallel and len(documents) > 3:
        return extractor.extract_metadata_parallel(documents)
    else:
        return extractor.extract_metadata(documents)


# Example usage
if __name__ == "__main__":
    # Create sample document
    sample_doc = Document(
        page_content="""
        Research Paper: Advanced RAG Systems
        
        Author: Dr. Jane Smith
        Organization: AI Research Institute
        Date: March 15, 2024
        
        Abstract: This paper explores advanced retrieval-augmented generation systems...
        
        Contact: jane.smith@airesearch.org
        DOI: 10.1234/airesearch.2024.001
        """,
        metadata={"source": "sample.pdf"}
    )
    
    # Extract metadata
    extractor = MetadataExtractor(use_llm=True)
    enriched_docs = extractor.extract_metadata([sample_doc])
    
    # Print results
    print("\n=== Extracted Metadata ===")
    print(enriched_docs[0].metadata)
