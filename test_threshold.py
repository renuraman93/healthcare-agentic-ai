"""
Prints the top-1 retrieval distance for on-topic and off-topic queries
so we can choose RELEVANCE_MAX_DISTANCE from real numbers.
"""

from app.rag.retriever import Retriever

QUERIES = {
    "ON-TOPIC": [
        "What medications is the patient taking?",
        "What is the patient's blood pressure?",
        "When is the follow-up appointment?",
    ],
    "OFF-TOPIC": [
        "What is the capital of France?",
        "How do I bake sourdough bread?",
        "Who won the last football world cup?",
    ],
    "HEALTH BUT NOT IN DOC": [
        "What is the patient's cholesterol level?",
    ],
}


def main() -> None:
    retriever = Retriever()
    for group, queries in QUERIES.items():
        print(f"\n--- {group} ---")
        for q in queries:
            top = retriever.retrieve(q, top_k=1)
            print(f"{top[0].relevance_distance:.4f}  {q}")


if __name__ == "__main__":
    main()