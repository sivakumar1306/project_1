import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
print("API Key set:", bool(key))

if key:
    llm = ChatGoogleGenerativeAI(google_api_key=key, model="gemini-2.5-flash")
    resp = llm.invoke("Hello, reply with 'Gemini 2.5 Flash is working!'")
    print("Response:", resp.content)
