"""Run from the project root:  python -m backend.test_speed
Measures how many seconds each Gemini model takes to answer one tiny question."""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)
client = genai.Client(
    api_key=os.getenv("API_KEY"),
    http_options=types.HttpOptions(timeout=60000),  # give up on any call after 60 seconds
)

models = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]

print("Testing speed (each model gets at most 60 seconds)...\n", flush=True)
for name in models:
    start = time.time()
    try:
        r = client.models.generate_content(
            model=name, contents="Reply with one short sentence: what is 2 + 2?"
        )
        print(f"WORKS  {name}: {time.time() - start:5.1f} s -> {r.text.strip()[:60]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED {name}: {time.time() - start:5.1f} s -> {str(e)[:120]}", flush=True)
print("\nDone.", flush=True)
