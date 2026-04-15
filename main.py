import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Carrega a chave de API do arquivo .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 2. Configura as "Leis" da IA (System Instructions)
instrucoes_sistema = """
Você é um assistente virtual focado estritamente e exclusivamente em análise de vendas e produtos.
Regra 1: Responda APENAS a perguntas relacionadas a vendas, produtos, faturamento e métricas comerciais.
Regra 2: Caso o usuário faça perguntas de saudações como 'Oi' e/ou 'tudo bem?', 'como está?', pode responder normalmente.
Regra 3: Se o usuário perguntar sobre qualquer outro assunto, recuse educadamente dizendo: 'Desculpe, meu escopo é restrito a dados de vendas e produtos. Como posso ajudar com seus resultados hoje?'.
Regra 4: Baseie suas respostas estritamente nos dados fornecidos no contexto.
Regra 5: De acordo com as perguntas do usuário, siga o ***DICIONARIO DE ENDPOINTS*** para saber qual endpoint complementar adicionar na requisição.



***DICIONARIO DE ENDPOINTS***:
/itens-vendidos/1/: traz pro usuário um ranking dos 7 itens mais vendidos por ele naquele periodo (24 horas) e informa a quantidade de cada um desses itens vendidos. 


"""



# Inicializa o modelo do Gemini com as instruções
model = genai.GenerativeModel(
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

# 3. Define a estrutura de dados que o React Native vai enviar
class RequisicaoChat(BaseModel):
    id_usuario: str
    pergunta: str
    token: str

@app.post("/api/chat")
async def chat_vendas(req: RequisicaoChat):
    try:
        # --- PASSO A: BUSCAR DADOS DO JAVA ---
        # Aqui você fará a requisição para o seu back-end em Java.
        # Descomente e ajuste as linhas abaixo quando a API Java estiver pronta:
        headers = {
            "Authorization": f"Bearer {req.token}"
        }
        url_java = f"http://localhost:8080/vendas/"
        resposta_java = requests.get(url_java, headers=headers)
        dados_vendas = resposta_java.json()
        
        # Para testar agora, vamos usar um dado simulado ("mock"):
        # dados_vendas = [
        #     {"id_produto": 101, "nome": "Teclado Mecânico", "quantidade_comprada": 2, "total_gasto": 350.00},
        #     {"id_produto": 102, "nome": "Mouse sem fio", "quantidade_comprada": 1, "total_gasto": 120.00}
        # ]

        # --- PASSO B: MONTAR O PROMPT COM CONTEXTO ---
        # Juntamos os dados do banco com a pergunta do usuário
        prompt_completo = f"""
        Aqui estão os dados de vendas deste usuário:
        {dados_vendas}
        
        Pergunta do usuário: {req.pergunta}
        """

        # --- PASSO C: ENVIAR PARA O GEMINI ---
        resposta_ia = model.generate_content(prompt_completo)

        # --- PASSO D: RETORNAR PARA O REACT NATIVE ---
        return {"resposta": resposta_ia.text}

    except Exception as e:
        # Se algo der errado, retorna erro 500 para o app não "crashar" no escuro
        raise HTTPException(status_code=500, detail=str(e))