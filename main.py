import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ==========================================
# 2A. INSTRUÇÕES DO ROTEADOR (IA para escolher a rota)
# ==========================================
instrucoes_roteador = """
Você é um classificador de intenções de usuário. 
Sua única função é ler a pergunta do usuário e retornar o ENDPOINT exato que deve ser consultado.
Regras:
- Se o usuário disser apenas uma saudação ("oi", "tudo bem") ou perguntar algo fora do escopo de vendas: retorne "NENHUM"
- Se o usuário perguntar sobre itens mais vendidos ou top produtos: retorne "/vendas/itens-vendidos/1"

Responda APENAS com a string do endpoint ou a palavra "NENHUM". Não adicione explicações ou aspas.
"""

model_router = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction=instrucoes_roteador,
    generation_config={"temperature": 0.0} 
)

# ==========================================
# 2B. INSTRUÇÕES DO ASSISTENTE FINAL
# ==========================================
instrucoes_sistema = """
Você é um assistente virtual focado estritamente e exclusivamente em análise de vendas e produtos.
Regra 1: Responda APENAS a perguntas relacionadas a vendas, produtos, faturamento e métricas comerciais.
Regra 2: Caso o usuário faça perguntas de saudações como 'Oi', 'tudo bem?', 'como está?', seja educado e responda normalmente.
Regra 3: Se o usuário perguntar sobre qualquer outro assunto, recuse educadamente dizendo: 'Desculpe, meu escopo é restrito a dados de vendas e produtos.'
Regra 4: Baseie suas respostas estritamente nos dados fornecidos no contexto.
"""

model_assistente = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction=instrucoes_sistema,
    generation_config={"temperature": 0.2}
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class RequisicaoChat(BaseModel):
    id_usuario: str
    pergunta: str
    token: str

@app.post("/api/chat")
async def chat_vendas(req: RequisicaoChat):
    try:
        resposta_rota = model_router.generate_content(req.pergunta)
        endpoint_escolhido = resposta_rota.text.strip()
        
        print(f"Endpoint escolhido pela IA: {endpoint_escolhido}") # 
        dados_vendas = {} 
        
        if endpoint_escolhido != "NENHUM":
            headers = {"Authorization": f"Bearer {req.token}"}
            url_java = f"http://localhost:8080{endpoint_escolhido}" 
            
            resposta_java = requests.get(url_java, headers=headers)
            dados_vendas = resposta_java.json()
            

        prompt_completo = f"""
        Aqui estão os dados do sistema correspondentes à solicitação do usuário:
        {dados_vendas}
        
        Pergunta do usuário: {req.pergunta}
        """

        resposta_ia = model_assistente.generate_content(prompt_completo)

        return {"resposta": resposta_ia.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))