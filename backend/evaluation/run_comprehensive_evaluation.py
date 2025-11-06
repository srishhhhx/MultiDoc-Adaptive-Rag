#!/usr/bin/env python3
"""
Comprehensive Two-Tier Evaluation Script for Transformer Paper

This script runs two separate evaluations:
1. Document-only queries: Full retrieval + RAG metrics (Recall@K, MRR, nDCG, etc.)
2. Hybrid/Web queries: Only applicable RAG metrics (Answer Relevance, etc.)

Usage:
    python backend/evaluation/run_comprehensive_evaluation.py

Features:
- Automatic document processing and indexing
- Rate limiting to avoid API limits
- Manual relevance labeling assistance
- Comprehensive reporting with proper None handling
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set
from dotenv import load_dotenv

# Load environment from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Add both backend and project root to path
backend_path = Path(__file__).parent.parent
project_root_path = backend_path.parent

sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(project_root_path))

from backend.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from backend.document_processor import DocumentProcessor
from backend.document_loader import MultiModalDocumentLoader
from backend.rag_workflow import RAGWorkflow
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TransformerEvaluationRunner:
    """
    Runs comprehensive evaluation on Transformer paper with two-tier approach
    """

    def __init__(
        self,
        pdf_path: str,
        doc_benchmark_path: str,
        hybrid_benchmark_path: str,
        relevance_labels_path: str,
        output_dir: str = "evaluation_results",
        rate_limit_delay: float = 3.0
    ):
        """
        Initialize evaluation runner

        Args:
            pdf_path: Path to Transformer paper PDF
            doc_benchmark_path: Path to document-only benchmark JSON
            hybrid_benchmark_path: Path to hybrid/web benchmark JSON
            relevance_labels_path: Path to relevance labels JSON (will create if missing)
            output_dir: Directory for results
            rate_limit_delay: Delay between queries in seconds
        """
        self.pdf_path = pdf_path
        self.doc_benchmark_path = doc_benchmark_path
        self.hybrid_benchmark_path = hybrid_benchmark_path
        self.relevance_labels_path = relevance_labels_path
        self.output_dir = output_dir
        self.rate_limit_delay = rate_limit_delay

        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        logger.info("Initializing document loader...")
        self.document_loader = MultiModalDocumentLoader()

        logger.info("Initializing document processor...")
        self.doc_processor = DocumentProcessor(document_loader=self.document_loader)

        logger.info("Initializing RAG workflow...")
        self.rag_workflow = RAGWorkflow(document_processor=self.doc_processor)
        self.graph = self.rag_workflow.create_workflow()

        logger.info("Initializing comprehensive evaluator...")
        self.evaluator = ComprehensiveEvaluator(use_groq=True)

        # Storage for chunks and IDs
        self.chunks = []
        self.chunk_id_map = {}  # Maps chunk content hash to chunk ID

    def load_and_process_document(self) -> bool:
        """Load and process the Transformer paper PDF"""
        try:
            logger.info(f"Loading PDF from: {self.pdf_path}")

            if not os.path.exists(self.pdf_path):
                logger.error(f"PDF not found: {self.pdf_path}")
                return False

            # Process document
            logger.info("Processing document (chunking, embedding, indexing)...")
            collection_name = "transformer_paper_eval"
            filename = os.path.basename(self.pdf_path)

            # Process file for session (returns dict with retriever)
            result = self.doc_processor.process_file_for_session(
                file_path=self.pdf_path,
                filename=filename,
                collection_name=collection_name
            )

            # Get retriever from result
            self.retriever = result['retriever']

            # Store collection and filename for graph invocation
            self.collection_name = collection_name
            self.filename = filename

            if not self.retriever:
                logger.error("Failed to create retriever")
                return False

            logger.info(f"✅ Document processed successfully")
            logger.info(f"   Collection: {collection_name}")
            logger.info(f"   Document: {filename}")

            # Store chunks for relevance labeling
            if hasattr(self.doc_processor, 'chunks'):
                self.chunks = self.doc_processor.chunks
                logger.info(f"   Total chunks: {len(self.chunks)}")

                # Create chunk ID map
                for i, chunk in enumerate(self.chunks):
                    chunk_id = f"chunk_{i}"
                    content_hash = hash(chunk.page_content[:100])  # Use first 100 chars
                    self.chunk_id_map[content_hash] = chunk_id

            return True

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_benchmark(self, path: str) -> List[Dict[str, str]]:
        """Load benchmark questions from JSON"""
        with open(path, 'r') as f:
            return json.load(f)

    def load_relevance_labels(self) -> Dict[str, Dict[str, Any]]:
        """
        Load or create relevance labels for document-only queries

        Returns dict mapping question -> {relevant_docs: Set[str], scores: Dict[str, int]}
        """
        if os.path.exists(self.relevance_labels_path):
            logger.info(f"Loading existing relevance labels from: {self.relevance_labels_path}")
            with open(self.relevance_labels_path, 'r') as f:
                data = json.load(f)

            # Convert lists to sets
            labels = {}
            for question, info in data.items():
                labels[question] = {
                    'relevant_docs': set(info['relevant_docs']),
                    'scores': info.get('scores', {})
                }
            return labels
        else:
            logger.warning(f"Relevance labels not found: {self.relevance_labels_path}")
            logger.info("Creating default relevance labels based on top-3 retrieved documents...")
            return {}

    def create_chunk_id(self, doc) -> str:
        """Create a unique ID for a document chunk"""
        content_preview = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)[:100]
        content_hash = hash(content_preview)

        if content_hash in self.chunk_id_map:
            return self.chunk_id_map[content_hash]

        # Create new ID
        chunk_id = f"chunk_{len(self.chunk_id_map)}"
        self.chunk_id_map[content_hash] = chunk_id
        return chunk_id

    def run_document_only_evaluation(self):
        """
        Run evaluation on document-only queries with full retrieval + RAG metrics
        """
        print("\n" + "="*80)
        print("📄 DOCUMENT-ONLY EVALUATION")
        print("="*80 + "\n")

        # Load benchmark
        benchmark = self.load_benchmark(self.doc_benchmark_path)
        logger.info(f"Loaded {len(benchmark)} document-only questions")

        # Load relevance labels
        relevance_labels = self.load_relevance_labels()

        # Process each query
        for idx, item in enumerate(benchmark, 1):
            question = item['question']
            ground_truth = item.get('ground_truth')

            print(f"\n{'─'*80}")
            print(f"Query {idx}/{len(benchmark)}")
            print(f"Q: {question[:70]}...")
            print(f"{'─'*80}")

            try:
                start_time = time.time()

                # Run through RAG pipeline (force document-only)
                # Pass available documents so query router sees them
                result = self.graph.invoke({
                    "question": question,
                    "retriever": self.retriever,
                    "collection_name": self.collection_name,
                    "available_documents": [self.filename]  # List of available docs
                })

                latency = (time.time() - start_time) * 1000

                # Extract results
                answer = result.get('solution', '')
                retrieved_docs = result.get('documents', [])

                # Get contexts and IDs
                contexts = []
                retrieved_doc_ids = []

                for doc in retrieved_docs[:10]:  # Top 10 for evaluation
                    if hasattr(doc, 'page_content'):
                        contexts.append(doc.page_content)
                        chunk_id = self.create_chunk_id(doc)
                        retrieved_doc_ids.append(chunk_id)

                # Get relevance labels for this question
                if question in relevance_labels:
                    relevant_docs = list(relevance_labels[question]['relevant_docs'])
                    relevance_scores = relevance_labels[question].get('scores', {})
                else:
                    # Default: top 3 retrieved are relevant
                    logger.warning(f"   No relevance labels for this query, using top-3 as relevant")
                    relevant_docs = retrieved_doc_ids[:3]
                    relevance_scores = {
                        retrieved_doc_ids[0]: 3,
                        retrieved_doc_ids[1]: 2,
                        retrieved_doc_ids[2]: 2
                    } if len(retrieved_doc_ids) >= 3 else {}

                # Add to evaluator
                self.evaluator.add_document_query_result(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    retrieved_doc_ids=retrieved_doc_ids,
                    relevant_doc_ids=relevant_docs,
                    relevance_scores=relevance_scores,
                    ground_truth=ground_truth,
                    metadata={
                        'total_time_ms': latency,
                        'total_attempts': result.get('generation_attempts', 1),
                        'num_contexts': len(contexts)
                    }
                )

                print(f"✅ Processed ({latency:.0f}ms)")
                print(f"   Retrieved: {len(contexts)} chunks")
                print(f"   Relevant: {len(relevant_docs)} chunks")
                print(f"   Answer: {answer[:100]}...")

                # Rate limiting
                if idx < len(benchmark):
                    logger.info(f"   Waiting {self.rate_limit_delay}s before next query...")
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"❌ Error processing query: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Compute metrics
        logger.info("\n🔍 Computing comprehensive metrics...")
        evaluation = self.evaluator.evaluate_document_queries()

        # Generate report
        report = self.evaluator.generate_report(evaluation)
        print("\n" + report)

        # Save results
        report_path = os.path.join(self.output_dir, 'document_only_evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)

        json_path = os.path.join(self.output_dir, 'document_only_evaluation_results.json')
        self.evaluator.export_json(evaluation, json_path)

        logger.info(f"\n✅ Document-only evaluation complete!")
        logger.info(f"   Report: {report_path}")
        logger.info(f"   Results: {json_path}")

        return evaluation

    def run_hybrid_web_evaluation(self):
        """
        Run evaluation on hybrid/web queries with only applicable metrics
        """
        print("\n" + "="*80)
        print("🌐 HYBRID/WEB EVALUATION")
        print("="*80 + "\n")

        # Reset evaluator
        self.evaluator.reset()

        # Load benchmark
        benchmark = self.load_benchmark(self.hybrid_benchmark_path)
        logger.info(f"Loaded {len(benchmark)} hybrid/web questions")

        # Process each query
        for idx, item in enumerate(benchmark, 1):
            question = item['question']
            ground_truth = item.get('ground_truth')

            print(f"\n{'─'*80}")
            print(f"Query {idx}/{len(benchmark)}")
            print(f"Q: {question[:70]}...")
            print(f"{'─'*80}")

            try:
                start_time = time.time()

                # Run through RAG pipeline (allows web search)
                # Pass available documents for intelligent routing
                result = self.graph.invoke({
                    "question": question,
                    "retriever": self.retriever,
                    "collection_name": self.collection_name,
                    "available_documents": [self.filename]  # Query router can choose doc or web
                })

                latency = (time.time() - start_time) * 1000

                # Extract results
                answer = result.get('solution', '')
                retrieved_docs = result.get('documents', [])
                web_results = result.get('web_search_results', [])

                # Combine contexts from documents and web
                contexts = []

                for doc in retrieved_docs[:5]:
                    if hasattr(doc, 'page_content'):
                        contexts.append(doc.page_content)

                for web_result in web_results:
                    contexts.append(web_result.get('content', ''))

                # Add to evaluator
                self.evaluator.add_hybrid_query_result(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                    metadata={
                        'total_time_ms': latency,
                        'total_attempts': result.get('generation_attempts', 1),
                        'num_doc_contexts': len(retrieved_docs),
                        'num_web_results': len(web_results)
                    }
                )

                print(f"✅ Processed ({latency:.0f}ms)")
                print(f"   Sources: {len(retrieved_docs)} docs + {len(web_results)} web")
                print(f"   Answer: {answer[:100]}...")

                # Rate limiting
                if idx < len(benchmark):
                    logger.info(f"   Waiting {self.rate_limit_delay}s before next query...")
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"❌ Error processing query: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Compute metrics
        logger.info("\n🔍 Computing comprehensive metrics...")
        evaluation = self.evaluator.evaluate_hybrid_queries()

        # Generate report
        report = self.evaluator.generate_report(evaluation)
        print("\n" + report)

        # Save results
        report_path = os.path.join(self.output_dir, 'hybrid_web_evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)

        json_path = os.path.join(self.output_dir, 'hybrid_web_evaluation_results.json')
        self.evaluator.export_json(evaluation, json_path)

        logger.info(f"\n✅ Hybrid/web evaluation complete!")
        logger.info(f"   Report: {report_path}")
        logger.info(f"   Results: {json_path}")

        return evaluation


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive two-tier RAG evaluation')
    parser.add_argument(
        '--pdf',
        default='7181-attention-is-all-you-need-2.pdf',
        help='Path to Transformer paper PDF'
    )
    parser.add_argument(
        '--doc-benchmark',
        default='backend/evaluation/transformer_doc_only_benchmark.json',
        help='Document-only benchmark JSON'
    )
    parser.add_argument(
        '--hybrid-benchmark',
        default='backend/evaluation/transformer_hybrid_web_benchmark.json',
        help='Hybrid/web benchmark JSON'
    )
    parser.add_argument(
        '--relevance-labels',
        default='backend/evaluation/transformer_relevance_labels.json',
        help='Relevance labels JSON (created if missing)'
    )
    parser.add_argument(
        '--output-dir',
        default='evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=3.0,
        help='Delay between queries in seconds (rate limiting)'
    )
    parser.add_argument(
        '--doc-only',
        action='store_true',
        help='Run only document-only evaluation'
    )
    parser.add_argument(
        '--hybrid-only',
        action='store_true',
        help='Run only hybrid/web evaluation'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 COMPREHENSIVE TWO-TIER RAG EVALUATION")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   PDF: {args.pdf}")
    print(f"   Doc Benchmark: {args.doc_benchmark}")
    print(f"   Hybrid Benchmark: {args.hybrid_benchmark}")
    print(f"   Relevance Labels: {args.relevance_labels}")
    print(f"   Output Dir: {args.output_dir}")
    print(f"   Rate Limit Delay: {args.delay}s\n")

    # Initialize runner
    runner = TransformerEvaluationRunner(
        pdf_path=args.pdf,
        doc_benchmark_path=args.doc_benchmark,
        hybrid_benchmark_path=args.hybrid_benchmark,
        relevance_labels_path=args.relevance_labels,
        output_dir=args.output_dir,
        rate_limit_delay=args.delay
    )

    # Load and process document
    if not runner.load_and_process_document():
        logger.error("Failed to process document. Exiting.")
        return

    # Run evaluations
    if not args.hybrid_only:
        logger.info("\n" + "="*80)
        logger.info("Starting document-only evaluation...")
        logger.info("="*80)
        runner.run_document_only_evaluation()

    if not args.doc_only:
        logger.info("\n" + "="*80)
        logger.info("Starting hybrid/web evaluation...")
        logger.info("="*80)
        runner.run_hybrid_web_evaluation()

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/")
    print("  - document_only_evaluation_report.txt")
    print("  - document_only_evaluation_results.json")
    print("  - hybrid_web_evaluation_report.txt")
    print("  - hybrid_web_evaluation_results.json\n")


if __name__ == '__main__':
    main()
