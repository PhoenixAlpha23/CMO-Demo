import os
from dotenv import load_dotenv
load_dotenv()

def load_env_vars():
    load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")