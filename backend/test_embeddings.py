import os

from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)
client = genai.Client(api_key=os.getenv("API_KEY"))

candidates = [
    "gemini-embedding-001",
    "gemini-embedding-2",
    "gemini-embedding-2-preview",
]

for name in candidates:
    try:
        r = client.models.embed_content(model=name, contents=["hello world"])
        print("WORKS :", name, "-> vector length", len(r.embeddings[0].values))
    except Exception as e:
        print("FAILED:", name, "->", str(e)[:200])
