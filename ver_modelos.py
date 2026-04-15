import os
import google.generativeai as genai
from dotenv import load_dotenv

# Carrega sua chave do .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Buscando modelos compatíveis...")

# Lista todos os modelos que suportam geração de texto
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"Nome exato para usar: {m.name}")