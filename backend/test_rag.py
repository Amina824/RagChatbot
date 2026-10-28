"""Run from the project root:  python -m backend.test_rag"""
import time

from backend.rag import RELEVANCE_THRESHOLD, generate_rag_response

questions = [
    "What are the office opening hours?",
    "How much does the Lakeside Villa cost?",
    "What is the seller commission?",
    "Are pets allowed in rentals?",
    "Who founded the company?",
    "What is the CEO's salary?",               # not in the documents
    "Do you have an office in Miami?",         # not in the documents
]

print(f"Relevance threshold: {RELEVANCE_THRESHOLD}", flush=True)
print(f"Running {len(questions)} questions. Please wait and do not open other terminals.\n", flush=True)

total_start = time.time()
for number, q in enumerate(questions, 1):
    start = time.time()
    r = generate_rag_response(q)
    seconds = time.time() - start
    dist = r["top_distance"]
    print(f"[{number}/{len(questions)}] Q: {q}", flush=True)
    print(f"   best distance: {dist:.3f}" if dist is not None else "   best distance: n/a", flush=True)
    print("   A:", r["answer"].replace("\n", " ")[:300], flush=True)
    print("   Sources:", r["sources"] or "none", "| grounded:", r["grounded"], flush=True)
    print(f"   (took {seconds:.1f} s)\n", flush=True)

print(f"All done in {time.time() - total_start:.1f} s", flush=True)
