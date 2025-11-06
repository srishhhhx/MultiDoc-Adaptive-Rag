"""
Comprehensive RAG Evaluation Framework

Combines:
1. Retrieval metrics (Recall@K, MRR, Precision@K, nDCG)
2. RAG quality metrics (Faithfulness, Answer Relevance, Context Precision/Recall)

Provides separate evaluation modes for:
- Document-only queries: Full retrieval + RAG metrics
- Hybrid/Web queries: Only applicable RAG metrics
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from evaluation.retrieval_metrics import RetrievalMetrics, RetrievalResult
from evaluation.ragas_evaluator import RAGPipelineEvaluator, EvaluationResult


@dataclass
class ComprehensiveEvaluation:
    """
    Complete evaluation results combining retrieval and generation metrics
    """
    # Query type classification
    query_type: str  # "document_only", "hybrid", or "web_only"
    num_queries: int

    # Always computed
    answer_relevance: float
    self_correction_rate: float
    avg_latency_ms: float
    ttft_ms: Optional[float]

    # Retrieval metrics (only for document-only queries)
    recall_at_5: Optional[float]
    recall_at_10: Optional[float]
    precision_at_5: Optional[float]
    precision_at_10: Optional[float]
    mrr: Optional[float]
    ndcg_at_5: Optional[float]
    ndcg_at_10: Optional[float]

    # RAG quality metrics
    faithfulness: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]

    # Additional metadata
    metadata: Dict[str, Any]


class ComprehensiveEvaluator:
    """
    Evaluator that combines retrieval and RAG metrics

    Usage for document-only queries:
        evaluator = ComprehensiveEvaluator()

        for query in benchmark:
            # Run through RAG pipeline
            result = rag_pipeline.process(query)

            # Add to evaluation
            evaluator.add_document_query_result(
                question=query['question'],
                answer=result['answer'],
                contexts=result['contexts'],
                retrieved_doc_ids=result['doc_ids'],
                relevant_doc_ids=query['relevant_docs'],  # Ground truth
                relevance_scores=query['relevance_scores'],  # Optional graded
                ground_truth=query.get('ground_truth'),
                metadata={'latency': result['latency']}
            )

        # Compute metrics
        evaluation = evaluator.evaluate_document_queries()
        print(evaluator.generate_report(evaluation))

    Usage for hybrid/web queries:
        evaluator = ComprehensiveEvaluator()

        for query in benchmark:
            result = rag_pipeline.process(query)

            evaluator.add_hybrid_query_result(
                question=query['question'],
                answer=result['answer'],
                contexts=result['contexts'],
                metadata={'latency': result['latency']}
            )

        evaluation = evaluator.evaluate_hybrid_queries()
        print(evaluator.generate_report(evaluation))
    """

    def __init__(self, use_groq: bool = True):
        """
        Initialize comprehensive evaluator

        Args:
            use_groq: Use Groq for RAGAs evaluation (cost-efficient)
        """
        self.rag_evaluator = RAGPipelineEvaluator(use_groq=use_groq)
        self.retrieval_evaluator = RetrievalMetrics()

        # Store query data for batch evaluation
        self.doc_queries = []
        self.hybrid_queries = []

    def add_document_query_result(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        relevance_scores: Optional[Dict[str, int]] = None,
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a document-only query result for evaluation

        Args:
            question: User's question
            answer: Generated answer
            contexts: Retrieved context chunks (text)
            retrieved_doc_ids: IDs of retrieved documents (in ranked order)
            relevant_doc_ids: Ground truth relevant document IDs
            relevance_scores: Optional graded relevance (doc_id -> score 0-3)
            ground_truth: Optional ground truth answer
            metadata: Optional metadata (latency, attempts, etc.)
        """
        self.doc_queries.append({
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'retrieved_doc_ids': retrieved_doc_ids,
            'relevant_doc_ids': set(relevant_doc_ids),
            'relevance_scores': relevance_scores or {},
            'ground_truth': ground_truth,
            'metadata': metadata or {}
        })

    def add_hybrid_query_result(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a hybrid/web query result for evaluation

        Args:
            question: User's question
            answer: Generated answer
            contexts: Retrieved/fetched context chunks
            ground_truth: Optional ground truth answer
            metadata: Optional metadata (latency, attempts, etc.)
        """
        self.hybrid_queries.append({
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'ground_truth': ground_truth,
            'metadata': metadata or {}
        })

    def evaluate_document_queries(self) -> ComprehensiveEvaluation:
        """
        Evaluate document-only queries with full retrieval + RAG metrics

        Returns:
            ComprehensiveEvaluation with all applicable metrics
        """
        if not self.doc_queries:
            raise ValueError("No document queries added. Use add_document_query_result() first.")

        print(f"\n{'='*70}")
        print(f"📊 Evaluating {len(self.doc_queries)} document-only queries")
        print(f"{'='*70}\n")

        # Extract data for RAGAs evaluation
        questions = [q['question'] for q in self.doc_queries]
        answers = [q['answer'] for q in self.doc_queries]
        contexts_list = [q['contexts'] for q in self.doc_queries]
        ground_truths = [q.get('ground_truth') for q in self.doc_queries]
        metadata_list = [q['metadata'] for q in self.doc_queries]

        # Has ground truth if at least one query has it
        has_ground_truth = any(gt for gt in ground_truths)

        # Evaluate retrieval quality
        print("🔍 Computing retrieval metrics...")
        for query in self.doc_queries:
            self.retrieval_evaluator.add_query_result(
                retrieved_docs=query['retrieved_doc_ids'],
                relevant_docs=query['relevant_doc_ids'],
                relevance_scores=query['relevance_scores']
            )

        retrieval_result = self.retrieval_evaluator.compute_metrics()
        print(f"   ✅ Recall@5: {retrieval_result.recall_at_5:.3f}")
        print(f"   ✅ Precision@5: {retrieval_result.precision_at_5:.3f}")
        print(f"   ✅ MRR: {retrieval_result.mrr:.3f}\n")

        # Evaluate RAG quality
        print("🤖 Computing RAG quality metrics...")
        rag_result = self.rag_evaluator.evaluate_batch(
            questions=questions,
            answers=answers,
            contexts_list=contexts_list,
            ground_truths=ground_truths if has_ground_truth else None,
            metadata_list=metadata_list
        )

        # Combine results
        evaluation = ComprehensiveEvaluation(
            query_type="document_only",
            num_queries=len(self.doc_queries),
            # Always computed
            answer_relevance=rag_result.answer_relevance,
            self_correction_rate=rag_result.self_correction_rate,
            avg_latency_ms=rag_result.avg_latency_ms,
            ttft_ms=rag_result.ttft_ms,
            # Retrieval metrics
            recall_at_5=retrieval_result.recall_at_5,
            recall_at_10=retrieval_result.recall_at_10,
            precision_at_5=retrieval_result.precision_at_5,
            precision_at_10=retrieval_result.precision_at_10,
            mrr=retrieval_result.mrr,
            ndcg_at_5=retrieval_result.ndcg_at_5,
            ndcg_at_10=retrieval_result.ndcg_at_10,
            # RAG quality
            faithfulness=rag_result.faithfulness,
            context_precision=rag_result.context_precision,
            context_recall=rag_result.context_recall,
            # Metadata
            metadata={
                'evaluation_time_s': rag_result.metadata['evaluation_time_s'],
                'num_relevant_docs': retrieval_result.num_relevant_docs,
                'has_ground_truth': has_ground_truth,
                'has_graded_relevance': retrieval_result.ndcg_at_5 is not None
            }
        )

        return evaluation

    def evaluate_hybrid_queries(self) -> ComprehensiveEvaluation:
        """
        Evaluate hybrid/web queries with only applicable metrics

        Returns:
            ComprehensiveEvaluation with retrieval metrics set to None
        """
        if not self.hybrid_queries:
            raise ValueError("No hybrid queries added. Use add_hybrid_query_result() first.")

        print(f"\n{'='*70}")
        print(f"📊 Evaluating {len(self.hybrid_queries)} hybrid/web queries")
        print(f"{'='*70}\n")

        # Extract data for RAGAs evaluation
        questions = [q['question'] for q in self.hybrid_queries]
        answers = [q['answer'] for q in self.hybrid_queries]
        contexts_list = [q['contexts'] for q in self.hybrid_queries]
        ground_truths = [q.get('ground_truth') for q in self.hybrid_queries]
        metadata_list = [q['metadata'] for q in self.hybrid_queries]

        has_ground_truth = any(gt for gt in ground_truths)

        # Evaluate RAG quality (only answer relevance for web queries)
        print("🤖 Computing RAG quality metrics...")
        rag_result = self.rag_evaluator.evaluate_batch(
            questions=questions,
            answers=answers,
            contexts_list=contexts_list,
            ground_truths=ground_truths if has_ground_truth else None,
            metadata_list=metadata_list
        )

        # Return evaluation with retrieval metrics as None
        evaluation = ComprehensiveEvaluation(
            query_type="hybrid_web",
            num_queries=len(self.hybrid_queries),
            # Always computed
            answer_relevance=rag_result.answer_relevance,
            self_correction_rate=rag_result.self_correction_rate,
            avg_latency_ms=rag_result.avg_latency_ms,
            ttft_ms=rag_result.ttft_ms,
            # Retrieval metrics (N/A for web/hybrid)
            recall_at_5=None,
            recall_at_10=None,
            precision_at_5=None,
            precision_at_10=None,
            mrr=None,
            ndcg_at_5=None,
            ndcg_at_10=None,
            # RAG quality (faithfulness may be N/A for web search)
            faithfulness=rag_result.faithfulness,
            context_precision=rag_result.context_precision,
            context_recall=rag_result.context_recall,
            # Metadata
            metadata={
                'evaluation_time_s': rag_result.metadata['evaluation_time_s'],
                'has_ground_truth': has_ground_truth
            }
        )

        return evaluation

    def generate_report(self, evaluation: ComprehensiveEvaluation) -> str:
        """
        Generate comprehensive evaluation report

        Args:
            evaluation: ComprehensiveEvaluation object

        Returns:
            Formatted report string
        """
        def fmt(value, spec=".3f"):
            return f"{value:{spec}}" if value is not None else "N/A"

        report = f"""
╔════════════════════════════════════════════════════════════════╗
║        COMPREHENSIVE RAG EVALUATION - DETAILED REPORT          ║
╚════════════════════════════════════════════════════════════════╝

📋 Evaluation Summary:
  • Query Type:                              {evaluation.query_type.upper()}
  • Total Queries:                           {evaluation.num_queries}
  • Evaluation Time:                         {evaluation.metadata['evaluation_time_s']:.2f}s
  • Average Latency per Query:               {evaluation.avg_latency_ms:.0f}ms

"""

        if evaluation.query_type == "document_only":
            report += f"""
🔍 RETRIEVAL METRICS (Document-Only):
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Recall (Coverage):
    • Recall@5:                              {fmt(evaluation.recall_at_5)}
    • Recall@10:                             {fmt(evaluation.recall_at_10)}

  Precision (Accuracy):
    • Precision@5:                           {fmt(evaluation.precision_at_5)}
    • Precision@10:                          {fmt(evaluation.precision_at_10)}

  Ranking Quality:
    • MRR (Mean Reciprocal Rank):            {fmt(evaluation.mrr)}
    • nDCG@5:                                {fmt(evaluation.ndcg_at_5)}
    • nDCG@10:                               {fmt(evaluation.ndcg_at_10)}

"""

        report += f"""
🤖 RAG QUALITY METRICS:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Answer Quality:
    • Answer Relevance:                      {fmt(evaluation.answer_relevance)}
    • Faithfulness (Hallucination Check):    {fmt(evaluation.faithfulness)}

  Context Quality:
    • Context Precision:                     {fmt(evaluation.context_precision)}
    • Context Recall:                        {fmt(evaluation.context_recall)}

⚙️  PIPELINE PERFORMANCE:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Efficiency:
    • Self-Correction Rate:                  {evaluation.self_correction_rate:.1%}
    • Average Latency:                       {evaluation.avg_latency_ms:.0f}ms
    • Time to First Token:                   {fmt(evaluation.ttft_ms, '.0f') + 'ms' if evaluation.ttft_ms else 'N/A'}

═══════════════════════════════════════════════════════════════

💡 INTERPRETATION GUIDE:

"""

        if evaluation.query_type == "document_only":
            report += """  Retrieval Metrics:
    ✅ Recall@5 > 0.7      → Good coverage of relevant documents
    ✅ Precision@5 > 0.6   → Most retrieved docs are relevant
    ✅ MRR > 0.5           → Most relevant doc appears early
    ✅ nDCG@5 > 0.7        → Good ranking quality

"""

        report += """  RAG Quality:
    ✅ Answer Relevance > 0.8   → Excellent query understanding
    ✅ Faithfulness > 0.7       → Low hallucination risk
    ✅ Context Precision > 0.7  → Good retrieval targeting
    ✅ Context Recall > 0.7     → Complete information retrieval

  Pipeline Performance:
    ✅ Self-Correction < 20%    → Efficient first-attempt generation
    ✅ Latency < 15s            → Good user experience

ℹ️  NOTE: N/A values indicate metrics not applicable for this evaluation type
          (e.g., retrieval metrics only apply to document-only queries)
"""

        return report

    def export_json(self, evaluation: ComprehensiveEvaluation, output_path: str):
        """Export evaluation results to JSON"""
        results = {
            'query_type': evaluation.query_type,
            'num_queries': evaluation.num_queries,
            'answer_relevance': evaluation.answer_relevance,
            'self_correction_rate': evaluation.self_correction_rate,
            'avg_latency_ms': evaluation.avg_latency_ms,
            'ttft_ms': evaluation.ttft_ms,
            'recall_at_5': evaluation.recall_at_5,
            'recall_at_10': evaluation.recall_at_10,
            'precision_at_5': evaluation.precision_at_5,
            'precision_at_10': evaluation.precision_at_10,
            'mrr': evaluation.mrr,
            'ndcg_at_5': evaluation.ndcg_at_5,
            'ndcg_at_10': evaluation.ndcg_at_10,
            'faithfulness': evaluation.faithfulness,
            'context_precision': evaluation.context_precision,
            'context_recall': evaluation.context_recall,
            'metadata': evaluation.metadata
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"💾 Results exported to: {output_path}")

    def reset(self):
        """Clear all stored queries"""
        self.doc_queries = []
        self.hybrid_queries = []
        self.retrieval_evaluator.reset()
        self.rag_evaluator.results_history = []
