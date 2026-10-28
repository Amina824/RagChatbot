import re
from pathlib import Path

from processing import SUPPORTED, process_document

folder = Path(__file__).resolve().parent.parent / "sample_docs"
all_chunks = {}
total = 0

for f in sorted(folder.iterdir()):
    if f.suffix.lower() in SUPPORTED:
        chunks = process_document(f)
        all_chunks[f.name] = chunks
        total += len(chunks)
        print(f"{f.name}: {len(chunks)} chunks")

print("\nTOTAL chunks:", total)

leftover = [
    c for chunks in all_chunks.values() for c in chunks
    if re.search(r"Meridian Crest Realty\s*\|", c["text"])
    or re.search(r"^Page \d+$", c["text"], re.M)
]
print("Footer lines left inside chunks:", len(leftover), "(should be 0)")

c = all_chunks["Property_Listings.pdf"][0]
print("\n--- Example chunk ---")
for key in ("chunk_id", "doc_id", "doc_name", "page", "chunk_index"):
    print(f"{key}: {c[key]}")
print("text:\n" + c["text"])
