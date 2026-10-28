"""RAG logic: retrieve relevant chunks, ask Gemini, and return a grounded answer with sources."""
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

import backend.vectorstore as vectorstore

load_dotenv(override=True)

# A call that takes longer than 30 s is abandoned so a backup model can answer instead.
client = genai.Client(api_key=os.getenv("API_KEY"),
                      http_options=types.HttpOptions(timeout=30000))

PRIMARY_MODEL = os.getenv("MODEL_NAME", "gemini-3.5-flash")
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]

# Cosine distance: lower = closer. This is a LOOSE cutoff that only blocks clearly unrelated
# questions. Borderline questions are handled by the model's "not found" rule below.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.55"))
CLOSE_MARGIN = 0.08   # fallback when the model does not report which blocks it used
TOP_K = 4

NOT_FOUND = "I could not find information about this in the documents."
BUSY = "The AI service is temporarily unavailable. Please try again in a moment."

SYSTEM_PROMPT = f"""You are an AI assistant for Meridian Crest Realty.
Answer the user's question using ONLY the numbered context blocks provided below.

STRICT RULES:
1. Use only the provided context. Do not use outside knowledge and do not guess.
2. If the context does not contain the answer, reply exactly: "{NOT_FOUND}"
3. If the context answers only part of the question, give that part and say what could not be found.
4. When a calculation is needed (for example a commission), use the numbers from the context and show the working briefly.
5. Keep answers clear, accurate and concise.
6. On the very last line, write which context blocks you actually used, in exactly this form:
   USED: 1, 3
   If you used none, write: USED: none
"""


class LLMUnavailable(Exception):
    """Raised when every model/retry attempt has failed."""


def call_llm_with_retry(prompt, retries=3, delay=2.0):
    """Call Gemini with retries on temporary errors, then fall back to other models."""
    models = [PRIMARY_MODEL] + [m for m in FALLBACK_MODELS if m != PRIMARY_MODEL]
    last_error = None
    for model in models:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                text = (response.text or "").strip()
                if text:
                    return text
                last_error = RuntimeError("empty response")
                break
            except Exception as e:  # noqa: BLE001
                last_error = e
                msg = str(e)
                if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
                    time.sleep(delay * (attempt + 1))  # temporary: wait, then retry
                else:
                    break  # permanent error or timeout: try the next model
    raise LLMUnavailable(str(last_error))


def _parse_used(answer):
    """Split the answer from its final 'USED: ...' line. Returns (answer, list_of_numbers or None)."""
    matches = list(re.finditer(r"(?im)^\W*USED\W*:\s*(none|[\d,\s]+?)\W*$", answer))
    if not matches:
        return answer.strip(), None
    m = matches[-1]
    body = answer[:m.start()].strip()
    raw = m.group(1).strip().lower()
    if raw == "none":
        return body, []
    return body, [int(n) for n in re.findall(r"\d+", raw)]


def _label(h):
    page = f"Page {h['page']}" if h["page"] else "no page number"
    return f"{h['doc_name']} ({page})"


def generate_rag_response(question, top_k=TOP_K):
    timings = {}
    start = time.time()
    hits = vectorstore.search(question, n_results=top_k)
    timings["search"] = round(time.time() - start, 2)
    top_distance = hits[0]["distance"] if hits else None

    # Guardrail 1: if nothing is even loosely related, do not ask the LLM at all.
    relevant = [h for h in hits if h["distance"] <= RELEVANCE_THRESHOLD]
    if not relevant:
        return {"answer": NOT_FOUND, "sources": [], "grounded": False,
                "top_distance": top_distance, "timings": timings}

    blocks = []
    for i, h in enumerate(relevant, 1):
        page = f"Page {h['page']}" if h["page"] else "no page number"
        blocks.append(f"[{i}] {h['doc_name']} | {page}\n{h['text']}")
    prompt = (f"{SYSTEM_PROMPT}\nCONTEXT:\n" + "\n\n---\n\n".join(blocks)
              + f"\n\nQUESTION: {question}")

    start = time.time()
    try:
        raw = call_llm_with_retry(prompt)
    except LLMUnavailable as e:
        print(f"[rag] LLM unavailable: {e}")
        timings["llm"] = round(time.time() - start, 2)
        return {"answer": BUSY, "sources": [], "grounded": False, "error": True,
                "top_distance": top_distance, "timings": timings}
    timings["llm"] = round(time.time() - start, 2)

    answer, used = _parse_used(raw)

    # Guardrail 2: the model decides whether the context really answered the question.
    close = [h for h in relevant if h["distance"] <= relevant[0]["distance"] + CLOSE_MARGIN]
    if used is None:      # the model ignored the USED format: fall back to the closest chunks
        grounded = bool(answer) and not answer.lower().startswith("i could not find information")
        cited = close
    else:
        grounded = bool(used) and bool(answer)
        cited = [relevant[n - 1] for n in used if 1 <= n <= len(relevant)] or close

    sources = []
    if grounded:
        for h in cited:
            if _label(h) not in sources:
                sources.append(_label(h))
    return {"answer": answer or NOT_FOUND, "sources": sources, "grounded": grounded,
            "top_distance": top_distance, "timings": timings}