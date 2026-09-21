# test_step6.py
import logging
from retrieval.hybrid_search import hybrid_search
from retrieval.confidence_scorer import score_results, format_source_citation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

queries = [
    "What is the resolution for error 944?",
    "How does the requisition approval process work?",
    "What happens when tax environment is missing?",
    "purple elephant dancing",  # Garbage query — should give low confidence
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    results = hybrid_search(query)
    score   = score_results(results)

    print(f"\nConfidence : {score['top_confidence']:.0%}")
    print(f"Band       : {score['confidence_band'].upper()}")
    print(f"Should answer   : {score['should_answer']}")
    print(f"Should escalate : {score['should_escalate']}")

    print(f"\nTop {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {format_source_citation(r)}")
        print(f"       {r['document'][:120]}...")

print("\n✅ Step 6 complete.")