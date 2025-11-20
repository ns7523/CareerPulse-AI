# list_models.py
import os, sys
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    sys.exit("ERROR: Set GOOGLE_API_KEY in .env before running this script")

genai.configure(api_key=API_KEY)

print("=== genai.list_models() OUTPUT ===")
try:
    for m in genai.list_models():
        print(" -", getattr(m, "name", getattr(m, "id", str(m))))
except Exception as e:
    print("genai.list_models() failed:", e)

print("\n=== client.models.list() OUTPUT (preferred) ===")
try:
    try:
        client = genai.Client(api_version="v1")
    except TypeError:
        client = genai.Client()
    models = client.models.list()
    for m in models:
        name = getattr(m, "name", None) or getattr(m, "id", None) or str(m)
        caps = getattr(m, "capabilities", None) or getattr(m, "supported_methods", None) or ""
        print(" -", name, "| capabilities:", caps)
except Exception as e:
    print("client.models.list() failed:", e)