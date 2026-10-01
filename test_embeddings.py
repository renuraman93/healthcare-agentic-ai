import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("API key found:", bool(api_key))
print("API key length:", len(api_key) if api_key else 0)

if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Patient has a history of hypertension."
)

print("Embedding API call successful!")
print("Embedding count:", len(response.embeddings))
print("Embedding dimensions:", len(response.embeddings[0].values))