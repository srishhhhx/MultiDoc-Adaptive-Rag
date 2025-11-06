"""
Retrieval-Specific Metrics for RAG Pipeline Evaluation

Implements standard information retrieval metrics:
- Recall@K: Percentage of relevant documents retrieved in top-K
- Precision@K: Percentage of top-K documents that are relevant
- MRR (Mean Reciprocal Rank): Position of first relevant document
- nDCG@K: Normalized Discounted Cumulative Gain for ranking quality

These metrics are only applicable to document-only queries (not web search).
"""

import math
from typing import List, Dict, Set, Optional
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Container for retrieval evaluation results"""
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    precision_at_10: float
    mrr: float
    ndcg_at_5: Optional[float]
    ndcg_at_10: Optional[float]
    num_queries: int
    num_relevant_docs: int


class RetrievalMetrics:
    """
    Information Retrieval metrics for evaluating document retrieval quality

    Usage:
        evaluator = RetrievalMetrics()

        # For each query, provide retrieved doc IDs and relevant doc IDs
        evaluator.add_query_result(
            retrieved_docs=['doc1', 'doc2', 'doc3', 'doc4', 'doc5'],
            relevant_docs={'doc2', 'doc5'},  # Ground truth relevant docs
            relevance_scores={'doc2': 3, 'doc5': 2}  # Optional graded scores
        )

        results = evaluator.compute_metrics()
    """

    def __init__(self):
        self.query_results = []

    def add_query_result(
        self,
        retrieved_docs: List[str],
        relevant_docs: Set[str],
        relevance_scores: Optional[Dict[str, int]] = None
    ):
        """
        Add a single query's retrieval results for evaluation

        Args:
            retrieved_docs: List of retrieved document IDs in ranked order
            relevant_docs: Set of ground-truth relevant document IDs
            relevance_scores: Optional dict mapping doc_id -> relevance score (0-3)
                             3=highly relevant, 2=relevant, 1=somewhat relevant, 0=not relevant
        """
        self.query_results.append({
            'retrieved': retrieved_docs,
            'relevant': relevant_docs,
            'scores': relevance_scores or {}
        })

    def recall_at_k(self, retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Recall@K: What percentage of relevant documents appear in top-K results?

        Formula: (# relevant docs in top-K) / (total # relevant docs)

        Range: 0.0 to 1.0 (higher is better)

        Example:
            Retrieved top-5: [doc1, doc2, doc3, doc4, doc5]
            Relevant docs: {doc2, doc5, doc7}
            Recall@5 = 2/3 = 0.667 (found 2 out of 3 relevant docs)
        """
        if not relevant_docs:
            return 0.0

        top_k = set(retrieved_docs[:k])
        found_relevant = len(top_k & relevant_docs)
        return found_relevant / len(relevant_docs)

    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Precision@K: What percentage of top-K results are relevant?

        Formula: (# relevant docs in top-K) / K

        Range: 0.0 to 1.0 (higher is better)

        Example:
            Retrieved top-5: [doc1, doc2, doc3, doc4, doc5]
            Relevant docs: {doc2, doc5}
            Precision@5 = 2/5 = 0.4 (2 out of 5 retrieved are relevant)
        """
        if k == 0:
            return 0.0

        top_k = set(retrieved_docs[:k])
        found_relevant = len(top_k & relevant_docs)
        return found_relevant / k

    def mrr(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """
        Mean Reciprocal Rank: Position of first relevant document

        Formula: 1 / (rank of first relevant doc)

        Range: 0.0 to 1.0 (higher is better)

        Example:
            Retrieved: [doc1, doc2, doc3, doc4, doc5]
            Relevant docs: {doc3, doc5}
            First relevant is doc3 at position 3
            MRR = 1/3 = 0.333

        Use case: Measures if the most relevant document appears early in results
        """
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / i
        return 0.0

    def dcg_at_k(self, retrieved_docs: List[str], relevance_scores: Dict[str, int], k: int) -> float:
        """
        Discounted Cumulative Gain at K

        Formula: DCG@K = Σ(rel_i / log2(i+1)) for i in 1 to K

        Gives higher weight to relevant docs appearing early in ranking.
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k], 1):
            relevance = relevance_scores.get(doc_id, 0)
            dcg += relevance / math.log2(i + 1)
        return dcg

    def ndcg_at_k(
        self,
        retrieved_docs: List[str],
        relevance_scores: Dict[str, int],
        k: int
    ) -> float:
        """
        Normalized Discounted Cumulative Gain at K

        Formula: nDCG@K = DCG@K / IDCG@K

        Range: 0.0 to 1.0 (higher is better)

        IDCG = Ideal DCG (DCG of perfect ranking by relevance scores)

        Example:
            Retrieved top-3: [doc1, doc2, doc3]
            Relevance scores: {doc1: 1, doc2: 3, doc3: 2}

            DCG@3 = 1/log2(2) + 3/log2(3) + 2/log2(4)
                  = 1.0 + 1.893 + 1.0 = 3.893

            Ideal ranking: [doc2, doc3, doc1] (sorted by relevance)
            IDCG@3 = 3/log2(2) + 2/log2(3) + 1/log2(4)
                   = 3.0 + 1.262 + 0.5 = 4.762

            nDCG@3 = 3.893 / 4.762 = 0.818

        Use case: Measures ranking quality when you have graded relevance (0-3 scale)
        """
        if not relevance_scores:
            return 0.0

        # Calculate DCG for retrieved ranking
        dcg = self.dcg_at_k(retrieved_docs, relevance_scores, k)

        # Calculate IDCG (ideal ranking)
        ideal_ranking = sorted(
            relevance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        ideal_docs = [doc_id for doc_id, _ in ideal_ranking]
        idcg = self.dcg_at_k(ideal_docs, relevance_scores, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def compute_metrics(self) -> RetrievalResult:
        """
        Compute aggregated metrics across all queries

        Returns:
            RetrievalResult with averaged metrics
        """
        if not self.query_results:
            raise ValueError("No query results added. Use add_query_result() first.")

        recall_5_scores = []
        recall_10_scores = []
        precision_5_scores = []
        precision_10_scores = []
        mrr_scores = []
        ndcg_5_scores = []
        ndcg_10_scores = []
        total_relevant = 0

        for result in self.query_results:
            retrieved = result['retrieved']
            relevant = result['relevant']
            scores = result['scores']

            recall_5_scores.append(self.recall_at_k(retrieved, relevant, 5))
            recall_10_scores.append(self.recall_at_k(retrieved, relevant, 10))
            precision_5_scores.append(self.precision_at_k(retrieved, relevant, 5))
            precision_10_scores.append(self.precision_at_k(retrieved, relevant, 10))
            mrr_scores.append(self.mrr(retrieved, relevant))

            total_relevant += len(relevant)

            # Only compute nDCG if relevance scores provided
            if scores:
                ndcg_5_scores.append(self.ndcg_at_k(retrieved, scores, 5))
                ndcg_10_scores.append(self.ndcg_at_k(retrieved, scores, 10))

        return RetrievalResult(
            recall_at_5=sum(recall_5_scores) / len(recall_5_scores),
            recall_at_10=sum(recall_10_scores) / len(recall_10_scores),
            precision_at_5=sum(precision_5_scores) / len(precision_5_scores),
            precision_at_10=sum(precision_10_scores) / len(precision_10_scores),
            mrr=sum(mrr_scores) / len(mrr_scores),
            ndcg_at_5=sum(ndcg_5_scores) / len(ndcg_5_scores) if ndcg_5_scores else None,
            ndcg_at_10=sum(ndcg_10_scores) / len(ndcg_10_scores) if ndcg_10_scores else None,
            num_queries=len(self.query_results),
            num_relevant_docs=total_relevant
        )

    def reset(self):
        """Clear all query results"""
        self.query_results = []


def create_relevance_labels_helper(
    question: str,
    retrieved_docs: List[str],
    doc_contents: List[str]
) -> Dict[str, int]:
    """
    Helper function to create relevance labels for a query

    This is a manual labeling interface. In practice, you would:
    1. Show the question
    2. Show each retrieved document
    3. Ask human labeler to assign relevance score 0-3

    For automated evaluation, you could use an LLM to assign scores.

    Args:
        question: The user's question
        retrieved_docs: List of document IDs
        doc_contents: List of document text contents (same order as IDs)

    Returns:
        Dict mapping doc_id -> relevance_score (0-3)
    """
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print(f"{'='*80}\n")

    labels = {}

    for doc_id, content in zip(retrieved_docs, doc_contents):
        print(f"\nDocument ID: {doc_id}")
        print(f"Content: {content[:300]}...")
        print("\nRelevance scoring:")
        print("  3 = Highly relevant (directly answers question)")
        print("  2 = Relevant (contains useful information)")
        print("  1 = Somewhat relevant (tangentially related)")
        print("  0 = Not relevant")

        while True:
            try:
                score = int(input("\nEnter relevance score (0-3): "))
                if 0 <= score <= 3:
                    labels[doc_id] = score
                    break
                else:
                    print("Please enter a number between 0 and 3")
            except ValueError:
                print("Please enter a valid number")

    return labels
