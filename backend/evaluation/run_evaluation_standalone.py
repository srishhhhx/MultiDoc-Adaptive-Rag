"""
Standalone Evaluation Script for Adaptive RAG Pipeline

This script:
1. Uploads a document and creates a session
2. Runs benchmark queries through the RAG pipeline
3. Evaluates using RAGAs metrics
4. Generates a comprehensive report

Usage:
    python backend/evaluation/run_evaluation_standalone.py \
        --document path/to/document.pdf \
        --benchmark backend/evaluation/benchmark.json \
        --output-dir evaluation_results
"""

import sys
import os
import json
import argparse
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
import uuid

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.ragas_evaluator import RAGPipelineEvaluator, EvaluationResult
from rag_workflow import RAGWorkflow
from session_manager import session_manager
from document_processor import DocumentProcessor
from document_loader import MultiModalDocumentLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionAwareRAGWorkflow(RAGWorkflow):
    """RAG workflow with session management support"""

    def __init__(self, document_processor=None):
        super().__init__(document_processor=document_processor)
        self.graph = None

    def _create_graph(self):
        """Create the graph instance"""
        return self.create_workflow()

    def get_graph(self):
        """Get or create the graph instance"""
        if self.graph is None:
            self.graph = self._create_graph()
        return self.graph

    def process_question_for_session(self, question: str, session_id: str):
        """Process a question through the RAG workflow for a specific session"""
        logger.info(f"Processing question for session {session_id}: {question}")

        # Get session retriever
        retriever = session_manager.get_session_retriever(session_id)
        if not retriever:
            raise ValueError("No documents found in session")

        # Get session data for collection_name
        session_data = session_manager.get_session(session_id)
        collection_name = session_data.collection_name

        # Create graph and invoke
        graph = self.get_graph()
        result = graph.invoke({
            "question": question,
            "retriever": retriever,
            "collection_name": collection_name
        })

        return result


def load_benchmark(benchmark_path: str) -> List[Dict[str, str]]:
    """Load benchmark dataset from JSON file"""
    with open(benchmark_path, 'r') as f:
        return json.load(f)


def upload_document(document_path: str) -> str:
    """Upload document and return session ID"""
    # Create new session
    session_id = session_manager.create_session()
    session_data = session_manager.get_session(session_id)
    collection_name = session_data.collection_name

    # Upload document
    doc_loader = MultiModalDocumentLoader()
    doc_processor = DocumentProcessor(doc_loader)

    filename = os.path.basename(document_path)
    result = doc_processor.process_file_for_session(document_path, filename, collection_name)

    # Set session retriever
    session_manager.set_session_retriever(session_id, result['retriever'])

    # Add document metadata to session
    file_metadata = {
        'filename': filename,
        'chunks': result['chunks'],
        'processed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'status': 'success',
        'document_id': f"doc_1_{filename}"
    }
    session_manager.add_document_to_session(session_id, file_metadata)

    return session_id


def run_queries_and_collect(
    session_id: str,
    benchmark: List[Dict[str, str]],
    rag_workflow: SessionAwareRAGWorkflow
) -> tuple[List[str], List[str], List[List[str]], List[str], List[Dict[str, Any]]]:
    """
    Run benchmark queries through the RAG pipeline and collect results

    Returns:
        Tuple of (questions, answers, contexts, ground_truths, metadata)
    """
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    metadata_list = []

    print(f"\n{'='*70}")
    print(f"🚀 Running {len(benchmark)} queries through RAG pipeline...")
    print(f"{'='*70}\n")

    for idx, item in enumerate(benchmark, 1):
        question = item['question']
        ground_truth = item.get('ground_truth', '')

        print(f"Query {idx}/{len(benchmark)}: {question[:80]}...")

        # Track timing
        start_time = time.time()

        try:
            # Run through RAG pipeline
            result = rag_workflow.process_question_for_session(question, session_id)

            total_time = (time.time() - start_time) * 1000  # Convert to ms

            # Extract results
            answer = result.get('solution', '')
            retrieved_docs = result.get('documents', [])
            contexts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retrieved_docs[:5]]

            # Collect metadata
            metadata = {
                'total_time_ms': total_time,
                'total_attempts': result.get('total_attempts', 1),
                'search_method': result.get('search_method', 'unknown'),
                'num_contexts': len(contexts)
            }

            questions.append(question)
            answers.append(answer)
            contexts_list.append(contexts)
            ground_truths.append(ground_truth)
            metadata_list.append(metadata)

            print(f"  ✅ Answer generated ({total_time:.0f}ms, {metadata['total_attempts']} attempt(s))")
            print(f"     Preview: {answer[:100]}...")
            print()

        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            logger.exception(f"Error processing query: {question}")
            continue

    return questions, answers, contexts_list, ground_truths, metadata_list


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG Pipeline with RAGAs (Standalone)')
    parser.add_argument('--document', required=True, help='Path to document to evaluate')
    parser.add_argument('--benchmark', required=True, help='Path to benchmark JSON')
    parser.add_argument('--output-dir', default='evaluation_results', help='Output directory for results')
    parser.add_argument('--use-groq', action='store_true', default=True, help='Use Groq for evaluation')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("  ADAPTIVE RAG PIPELINE - EVALUATION FRAMEWORK (STANDALONE)")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"   Document: {args.document}")
    print(f"   Benchmark: {args.benchmark}")
    print(f"   Output Dir: {args.output_dir}")
    print(f"   Use Groq: {args.use_groq}")

    # Load benchmark
    try:
        benchmark = load_benchmark(args.benchmark)
        print(f"\n✅ Loaded {len(benchmark)} test questions from benchmark")
    except Exception as e:
        print(f"\n❌ Error loading benchmark: {e}")
        return

    # Upload document
    print(f"\n📤 Uploading document...")
    try:
        session_id = upload_document(args.document)
        session_info = session_manager.get_session_info(session_id)
        print(f"✅ Document uploaded (Session: {session_id})")
        print(f"   Documents: {len(session_info['documents'])}")
        print(f"   Chunks: {session_info['documents'][0]['chunks']}")
    except Exception as e:
        print(f"\n❌ Error uploading document: {e}")
        logger.exception("Document upload failed")
        return

    # Initialize RAG workflow
    rag_workflow = SessionAwareRAGWorkflow()

    # Run queries and collect results
    questions, answers, contexts_list, ground_truths, metadata_list = run_queries_and_collect(
        session_id,
        benchmark,
        rag_workflow
    )

    if not questions:
        print("\n❌ No successful queries to evaluate")
        return

    print(f"\n{'='*70}")
    print(f"✅ Collected {len(questions)} successful responses")
    print(f"{'='*70}\n")

    # Initialize evaluator
    print("🔍 Initializing RAGAs evaluator...")
    evaluator = RAGPipelineEvaluator(use_groq=args.use_groq)

    # Run evaluation
    try:
        result = evaluator.evaluate_batch(
            questions=questions,
            answers=answers,
            contexts_list=contexts_list,
            ground_truths=ground_truths if ground_truths else None,
            metadata_list=metadata_list
        )

        # Generate and display report
        print("\n" + "="*70)
        report = evaluator.generate_report(result)
        print(report)

        # Save report
        report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)

        # Export JSON results
        json_path = os.path.join(args.output_dir, 'evaluation_results.json')
        evaluator.export_results_json(json_path)

        print(f"\n✅ Evaluation complete!")
        print(f"   Report: {report_path}")
        print(f"   Results: {json_path}")
        print(f"\n📊 Quick Summary:")
        print(f"   Faithfulness: {result.faithfulness:.3f}")
        print(f"   Answer Relevance: {result.answer_relevance:.3f}")
        print(f"   Context Precision: {result.context_precision:.3f}")
        print(f"   Context Recall: {result.context_recall:.3f}")
        print(f"   Self-Correction Rate: {result.self_correction_rate:.1%}")
        print(f"   Avg Latency: {result.avg_latency_ms:.0f}ms")

        # Clean up session
        session_manager.delete_session(session_id)
        print(f"\n🧹 Cleaned up session {session_id}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        logger.exception("Evaluation failed")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
