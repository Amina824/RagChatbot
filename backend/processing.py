"""Document processing: extract text, clean it, and split it into chunks with metadata."""
import re
import uuid
from pathlib import Path

from docx import Document
from pypdf import PdfReader

CHUNK_SIZE = 600      # maximum characters per chunk
CHUNK_OVERLAP = 100   # characters repeated between neighbouring chunks
SUPPORTED = {".pdf", ".docx", ".txt"}


# ---------- 1. Extraction ----------
def extract_pages(path):
    """Return a list of (page_number, text). page_number is None when the format has no pages."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    if ext == ".docx":
        doc = Document(str(path))
        lines = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                lines.append(" | ".join(c.text.strip() for c in row.cells))
        return [(None, "\n".join(lines))]
    if ext == ".txt":
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {ext}")


# ---------- 2. Cleaning ----------
def _norm(line):
    """Normalise a line so 'Page 1' and 'Page 2' look the same."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def remove_headers_footers(pages):
    """Remove lines repeated at the top/bottom of every page (headers, footers, page numbers)."""
    if len(pages) < 2:
        return pages
    all_lines = [[l for l in text.splitlines() if l.strip()] for _, text in pages]
    edge_sets = [{_norm(l) for l in lines[:3] + lines[-3:]} for lines in all_lines]
    repeated = set.intersection(*edge_sets)
    cleaned = []
    for (num, _), lines in zip(pages, all_lines):
        keep = []
        for i, line in enumerate(lines):
            at_edge = i < 3 or i >= len(lines) - 3
            if at_edge and _norm(line) in repeated:
                continue
            keep.append(line)
        cleaned.append((num, "\n".join(keep)))
    return cleaned


def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


# ---------- 3. Chunking ----------
def _split_long(line, size):
    """Split a very long line at sentence ends (hard cut as a last resort)."""
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", line):
        if len(s) > size:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(s[i:i + size] for i in range(0, len(s), size))
            continue
        if cur and len(cur) + len(s) + 1 > size:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Group whole lines into chunks of up to `size` characters, with line-level overlap."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    chunks, current, length = [], [], 0
    for line in lines:
        pieces = [line] if len(line) <= size else _split_long(line, size)
        for piece in pieces:
            if current and length + len(piece) + 1 > size:
                chunks.append("\n".join(current))
                tail, tail_len = [], 0
                for l in reversed(current):
                    if tail_len + len(l) + 1 > overlap:
                        break
                    tail.insert(0, l)
                    tail_len += len(l) + 1
                current, length = tail, tail_len
            current.append(piece)
            length += len(piece) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# ---------- 4. Full pipeline for one document ----------
def process_document(path, doc_id=None, doc_name=None):
    """Extract -> clean -> chunk. Returns a list of chunk dicts that carry metadata."""
    path = Path(path)
    doc_id = doc_id or uuid.uuid4().hex[:8]
    doc_name = doc_name or path.name

    pages = remove_headers_footers(extract_pages(path))
    chunks, idx = [], 0
    for page_num, text in pages:
        text = clean_text(text)
        if not text:
            continue
        for piece in chunk_text(text):
            chunks.append({
                "chunk_id": f"{doc_id}-{idx:03d}",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page": page_num,
                "chunk_index": idx,
                "text": piece,
            })
            idx += 1
    if not chunks:
        raise ValueError("No readable text found in this document (it may be a scanned image).")
    return chunks
