"""
RAGAs-based Evaluation Framework for Adaptive RAG Pipeline

Metrics implemented:
1. Faithfulness - Measures factual accuracy against retrieved context
2. Answer Relevance - Measures how well answer addresses the question
3. Context Precision - Measures quality of retrieved chunks
4. Context Recall - Measures completeness of retrieved information

Uses Groq (Llama3) for cost-efficient evaluation.
"""

import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️ RAGAs not installed. Run: pip install ragas")


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    faithfulness: Optional[float]  # None when web search used or N/A
    answer_relevance: float
    context_precision: Optional[float]  # None when ground truth missing
    context_recall: Optional[float]  # None when ground truth missing
    self_correction_rate: float
    avg_latency_ms: float
    ttft_ms: Optional[float]
    metadata: Dict[str, Any]


class RAGPipelineEvaluator:
    """
    Comprehensive evaluator for the Adaptive RAG Pipeline

    Combines RAGAs metrics with custom pipeline-specific metrics.
    """

    def __init__(self, use_groq: bool = True):
        """
        Initialize evaluator

        Args:
            use_groq: Use Groq for evaluation (cost-efficient) instead of OpenAI
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAs is required. Install with: pip install ragas")

        self.use_groq = use_groq
        self.results_history = []

        # Verify API key is set if using Groq
        if use_groq and not os.getenv('GROQ_API_KEY'):
            raise ValueError(
                "GROQ_API_KEY not found in environment. "
                "Please set it in your .env file or environment variables."
            )

    def evaluate_single_query(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single query-answer pair

        Args:
            question: User's question
            answer: Generated answer
            contexts: Retrieved context chunks
            ground_truth: Expected answer (if available)
            metadata: Additional metadata (latency, attempts, etc.)

        Returns:
            Dictionary with metric scores
        """
        # Prepare data for RAGAs
        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [contexts],
        }

        if ground_truth:
            data['ground_truth'] = [ground_truth]

        dataset = Dataset.from_dict(data)

        # Select metrics based on available data
        metrics_to_use = [faithfulness, answer_relevancy, context_precision]
        if ground_truth:
            metrics_to_use.append(context_recall)

        try:
            # Run evaluation
            result = evaluate(
                dataset,
                metrics=metrics_to_use,
                llm=self._get_llm() if self.use_groq else None,
                embeddings=self._get_embeddings() if self.use_groq else None
            )

            # RAGAs returns lists of scores, take the first (single query)
            scores = {
                'faithfulness': result['faithfulness'][0],
                'answer_relevance': result['answer_relevancy'][0],
                'context_precision': result['context_precision'][0],
            }

            if ground_truth:
                scores['context_recall'] = result['context_recall'][0]

            # Add custom metrics from metadata
            if metadata:
                if 'total_attempts' in metadata:
                    scores['self_correction_rate'] = 1.0 if metadata['total_attempts'] > 1 else 0.0
                if 'total_time_ms' in metadata:
                    scores['latency_ms'] = metadata['total_time_ms']
                if 'ttft_ms' in metadata:
                    scores['ttft_ms'] = metadata['ttft_ms']

            return scores

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {}

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> EvaluationResult:
        """
        Evaluate multiple query-answer pairs and aggregate results

        Args:
            questions: List of questions
            answers: List of generated answers
            contexts_list: List of context lists
            ground_truths: List of expected answers (optional)
            metadata_list: List of metadata dicts (optional)

        Returns:
            EvaluationResult with aggregated metrics
        """
        # Prepare batch data
        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts_list,
        }

        if ground_truths:
            data['ground_truth'] = ground_truths

        dataset = Dataset.from_dict(data)

        # Select metrics
        metrics_to_use = [faithfulness, answer_relevancy, context_precision]
        if ground_truths:
            metrics_to_use.append(context_recall)

        try:
            # Run batch evaluation
            print(f"🔍 Evaluating {len(questions)} queries...")
            start_time = time.time()

            result = evaluate(
                dataset,
                metrics=metrics_to_use,
                llm=self._get_llm() if self.use_groq else None,
                embeddings=self._get_embeddings() if self.use_groq else None
            )

            eval_time = time.time() - start_time
            print(f"✅ Evaluation complete ({eval_time:.2f}s)")

            # Convert to pandas and calculate means
            df = result.to_pandas()

            # Handle metrics that may not be present (use None instead of 0)
            faithfulness_score = None
            context_precision_score = None
            context_recall_score = None

            if 'faithfulness' in df.columns:
                faithfulness_score = float(df['faithfulness'].mean())

            if 'answer_relevancy' in df.columns:
                answer_relevance_score = float(df['answer_relevancy'].mean())
            else:
                answer_relevance_score = 0.0  # This should always be present

            if 'context_precision' in df.columns:
                context_precision_score = float(df['context_precision'].mean())

            if 'context_recall' in df.columns:
                context_recall_score = float(df['context_recall'].mean())

            # Calculate custom metrics
            self_correction_rate = 0.0
            avg_latency = 0.0
            avg_ttft = None

            if metadata_list:
                corrections = sum(1 for m in metadata_list if m.get('total_attempts', 1) > 1)
                self_correction_rate = corrections / len(metadata_list)

                latencies = [m.get('total_time_ms', 0) for m in metadata_list if 'total_time_ms' in m]
                if latencies:
                    avg_latency = sum(latencies) / len(latencies)

                ttfts = [m.get('ttft_ms', 0) for m in metadata_list if 'ttft_ms' in m]
                if ttfts:
                    avg_ttft = sum(ttfts) / len(ttfts)

            # Create result object
            eval_result = EvaluationResult(
                faithfulness=faithfulness_score,
                answer_relevance=answer_relevance_score,
                context_precision=context_precision_score,
                context_recall=context_recall_score,
                self_correction_rate=self_correction_rate,
                avg_latency_ms=avg_latency,
                ttft_ms=avg_ttft,
                metadata={
                    'num_queries': len(questions),
                    'evaluation_time_s': eval_time,
                    'used_ground_truth': ground_truths is not None
                }
            )

            # Store in history
            self.results_history.append(eval_result)

            return eval_result

        except Exception as e:
            print(f"❌ Batch evaluation failed: {e}")
            raise

    def _get_llm(self):
        """Get LLM for RAGAs evaluation (using Groq)"""
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    def _get_embeddings(self):
        """Get embeddings for RAGAs evaluation"""
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def generate_report(self, result: EvaluationResult, output_path: Optional[str] = None) -> str:
        """
        Generate a formatted evaluation report

        Args:
            result: EvaluationResult to report on
            output_path: Optional path to save report

        Returns:
            Formatted report string
        """
        # Format values with N/A for None
        def format_metric(value, format_spec=".3f"):
            if value is None:
                return "N/A"
            return f"{value:{format_spec}}"

        report = f"""
╔════════════════════════════════════════════════════════════════╗
║           ADAPTIVE RAG PIPELINE - EVALUATION REPORT            ║
╚════════════════════════════════════════════════════════════════╝

📊 RAGAs Quality Metrics:
  • Faithfulness (Hallucination Detection):  {format_metric(result.faithfulness)}
  • Answer Relevance:                        {format_metric(result.answer_relevance)}
  • Context Precision (Retrieval Quality):   {format_metric(result.context_precision)}
  • Context Recall (Retrieval Completeness): {format_metric(result.context_recall)}

⚙️  Pipeline-Specific Metrics:
  • Self-Correction Rate:                    {result.self_correction_rate:.1%}
  • Average End-to-End Latency:              {result.avg_latency_ms:.0f}ms
  • Average Time to First Token (TTFT):      {format_metric(result.ttft_ms, '.0f') + 'ms' if result.ttft_ms else 'N/A'}

📈 Metadata:
  • Queries Evaluated:                       {result.metadata['num_queries']}
  • Evaluation Time:                         {result.metadata['evaluation_time_s']:.2f}s
  • Used Ground Truth:                       {result.metadata['used_ground_truth']}

═══════════════════════════════════════════════════════════════

💡 Interpretation Guide:
  • Faithfulness > 0.7:    Low hallucination risk
  • Answer Relevance > 0.8: High query understanding
  • Context Precision > 0.7: Good retrieval quality
  • Context Recall > 0.7:   Complete information retrieval
  • Self-Correction < 20%:  Efficient first-attempt generation
  • TTFT < 2000ms:          Good streaming performance

ℹ️  Note: N/A values indicate metrics not applicable for this evaluation
           (e.g., Context Precision/Recall require ground truth answers)
"""

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to: {output_path}")

        return report

    def export_results_json(self, output_path: str):
        """Export all evaluation results to JSON"""
        results_dict = [
            {
                'faithfulness': r.faithfulness,
                'answer_relevance': r.answer_relevance,
                'context_precision': r.context_precision,
                'context_recall': r.context_recall,
                'self_correction_rate': r.self_correction_rate,
                'avg_latency_ms': r.avg_latency_ms,
                'ttft_ms': r.ttft_ms,
                'metadata': r.metadata
            }
            for r in self.results_history
        ]

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"💾 Results exported to: {output_path}")


def create_benchmark_dataset(documents_path: str, num_questions: int = 20) -> List[Dict[str, Any]]:
    """
    Helper function to create a benchmark dataset from documents

    Args:
        documents_path: Path to documents
        num_questions: Number of test questions to generate

    Returns:
        List of benchmark question-answer pairs
    """
    # This would use an LLM to generate test Q&A pairs from documents
    # Placeholder implementation
    print(f"⚠️ Benchmark dataset generation not implemented yet")
    print(f"   Manually create {num_questions} test questions for now")
    return []
