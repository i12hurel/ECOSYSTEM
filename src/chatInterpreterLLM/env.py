from dotenv import load_dotenv
import os

def enviroment():
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")
    return API_KEY