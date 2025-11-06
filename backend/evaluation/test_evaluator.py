"""
Quick test script to verify RAGAs evaluation framework

This runs a minimal test without needing a full session.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.ragas_evaluator import RAGPipelineEvaluator


def test_basic_evaluation():
    """Test basic evaluation functionality"""
    print("\n" + "="*70)
    print("  Testing RAGAs Evaluation Framework")
    print("="*70 + "\n")

    # Sample data
    questions = [
        "What is supply chain risk management?",
        "How does Fuzzy AHP work?"
    ]

    answers = [
        "Supply chain risk management involves identifying, assessing, and mitigating risks in the supply chain using various methodologies including Fuzzy AHP and TOPSIS.",
        "Fuzzy AHP uses fuzzy logic to handle uncertainty in pairwise comparisons, converting linguistic judgments into numerical values for hierarchical decision making."
    ]

    contexts_list = [
        [
            "Supply chain risk management is crucial for business continuity. It involves systematic identification of potential risks.",
            "Various methodologies exist for risk assessment, including Fuzzy AHP which handles uncertainty effectively."
        ],
        [
            "Fuzzy AHP extends traditional AHP by incorporating fuzzy set theory.",
            "Pairwise comparisons in Fuzzy AHP use triangular fuzzy numbers to represent uncertainty."
        ]
    ]

    ground_truths = [
        "Supply chain risk management is the process of identifying, assessing, and mitigating risks that could disrupt the supply chain.",
        "Fuzzy AHP is a decision-making method that combines fuzzy set theory with the Analytic Hierarchy Process to handle uncertainty in expert judgments."
    ]

    metadata_list = [
        {'total_attempts': 1, 'total_time_ms': 2340},
        {'total_attempts': 2, 'total_time_ms': 3120}  # This one required self-correction
    ]

    try:
        # Initialize evaluator
        print("🔍 Initializing RAGAs evaluator...")
        evaluator = RAGPipelineEvaluator(use_groq=True)
        print("✅ Evaluator initialized\n")

        # Run evaluation
        print(f"📊 Evaluating {len(questions)} sample queries...")
        result = evaluator.evaluate_batch(
            questions=questions,
            answers=answers,
            contexts_list=contexts_list,
            ground_truths=ground_truths,
            metadata_list=metadata_list
        )

        # Generate report
        report = evaluator.generate_report(result)
        print(report)

        print("\n✅ Test successful! Evaluation framework is working correctly.")
        print("\nNext steps:")
        print("  1. Upload your document to get a session ID")
        print("  2. Run: python backend/evaluation/run_evaluation.py --session-id <your-session-id>")
        print("  3. View the comprehensive evaluation report\n")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_basic_evaluation()
    sys.exit(0 if success else 1)
