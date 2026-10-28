from pathlib import Path

from processing import SUPPORTED, process_document
import vectorstore

folder = Path(__file__).resolve().parent.parent / "sample_docs"

# 1. Start from an empty database, then index all sample documents
vectorstore.reset_collection()
total = 0
for f in sorted(folder.iterdir()):
    if f.suffix.lower() in SUPPORTED:
        total += vectorstore.add_chunks(process_document(f))
print(f"Indexed {total} chunks. Chunks in database: {vectorstore.count_chunks()}\n")

# 2. Try some questions
questions = [
    "What are the office opening hours?",
    "How much does the Lakeside Villa cost?",
    "What is the seller commission?",
    "Are pets allowed in rentals?",
    "What is the CEO's salary?",          # not in the documents on purpose
]
for q in questions:
    print("Q:", q)
    for h in vectorstore.search(q, n_results=3):
        snippet = h["text"].replace("\n", " ")[:70]
        print(f"   dist={h['distance']:.3f} | {h['doc_name']} p.{h['page']} | {snippet}...")
    print()
