import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("API_KEY"))

candidates = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]

for name in candidates:
    try:
        r = client.models.generate_content(model=name, contents="Say hi in 3 words.")
        print("WORKS  :", name, "->", r.text.strip())
    except Exception as e:
        print("FAILED :", name, "->", str(e)[:200])