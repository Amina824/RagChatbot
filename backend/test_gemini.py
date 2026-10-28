import os
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME")

client = genai.Client(api_key=api_key)

if not model_name or model_name == "to_be_set":
    print("Gemini models available for your key:\n")
    for m in client.models.list():
        if "gemini" in m.name.lower():
            print(m.name.replace("models/", ""))
    print("\nChoose ONE name, set MODEL_NAME=<name> in .env, then run this again.")
else:
    response = client.models.generate_content(
        model=model_name,
        contents="Say hello in one short sentence.",
    )
    print("Model replied:", response.text)