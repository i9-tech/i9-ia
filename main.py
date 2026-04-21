import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import date, timedelta
import jwt

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ==========================================
# ROTEADOR — escolhe qual endpoint chamar
# ==========================================
instrucoes_roteador = """
Você é um classificador de intenções de usuário.
Sua única função é ler a pergunta do usuário e retornar o ENDPOINT exato que deve ser consultado.

DICIONÁRIO DE ENDPOINTS (substitua {empresaId} pelo valor informado no contexto):
/vendas/{id} — usuário pergunta sobre uma venda específica pelo ID.
/vendas/lucro-total?idFuncionario={idFuncionario} — usuário pergunta sobre lucro total de um funcionário.
/vendas/valor-total-diario/{empresaId} — usuário quer saber o faturamento bruto total da empresa.
/vendas/valor-liquido-diario/{empresaId} — usuário pergunta sobre lucro líquido da empresa.
/vendas/valor-total-setor-diario/{empresaId} — usuário quer comparar vendas por setor hoje.
/vendas/valor-total-categoria-diario/{empresaId} — usuário quer comparar vendas por categoria hoje.
/vendas/quantidade-vendas/{empresaId} — usuário pergunta quantas vendas foram feitas.
/vendas/quantidade-minima/{empresaId} — usuário pergunta sobre produtos com estoque baixo.
/vendas/itens-vendidos/{empresaId} — usuário quer saber quais itens foram vendidos hoje.
/vendas/pratos-vendidos/{empresaId} — usuário quer ver lista detalhada de pratos vendidos.
/vendas/top-pratos/{empresaId} — usuário pergunta sobre pratos mais vendidos ou ranking de pratos.
/vendas/top-produtos/{empresaId} — usuário pergunta sobre produtos mais vendidos ou ranking de produtos.
/vendas/top-categorias/{empresaId} — usuário quer saber quais categorias mais venderam.
/vendas/ranking-setores/{empresaId} — usuário quer comparar setores por desempenho.
/vendas/kpis/{empresaId} — usuário pede indicadores, métricas gerais ou resumo de desempenho.

Regras:
- SEMPRE substitua {empresaId} e {idFuncionario} pelos valores informados no contexto.
- Se mencionar 'semana', adicione ?dataInicio={inicio_semana}&dataFim={hoje} ao endpoint.
- Se mencionar 'mês', adicione ?dataInicio={inicio_mes}&dataFim={hoje} ao endpoint.
- Se for saudação ou assunto fora de vendas: retorne "NENHUM".
- Responda APENAS com a string do endpoint ou "NENHUM". Sem explicações ou aspas.
"""

model_router = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction=instrucoes_roteador,
    generation_config={"temperature": 0.0}
)

# ==========================================
# ASSISTENTE — responde ao usuário
# ==========================================
instrucoes_sistema = """
Você é um assistente virtual focado estritamente e exclusivamente em análise de vendas e produtos.
Regra 1: Responda APENAS a perguntas relacionadas a vendas, produtos, faturamento e métricas comerciais.
Regra 2: Caso o usuário faça perguntas de saudações como 'Oi', 'tudo bem?', 'como está?', responda normalmente e de forma breve.
Regra 3: Se o usuário perguntar sobre qualquer outro assunto, recuse educadamente dizendo: 'Desculpe, meu conhecimento é restrito a dados de vendas e gerenciamento de produtos. Como posso ajudar com seus resultados hoje?'
Regra 4: Baseie suas respostas exclusivamente nos dados fornecidos no contexto. Nunca invente valores.
Regra 5: Nunca exiba endpoints, URLs, parâmetros técnicos ou nomes de API na resposta.
Regra 6: Responda de forma clara, direta e amigável, como um assistente conversacional.
Regra 7: Pratos são itens do cardápio (pastel, lanche, prato feito, suco, etc). Produtos são itens de estoque (salgadinhos, sorvetes, doces, etc). Nunca os confunda.
Regra 8: Sempre formate valores monetários em reais com o símbolo R$ e duas casas decimais usando vírgula. Exemplo: R$ 528,72 em vez de 528.72.
Regra 9: Nunca use markdown na resposta. Não use asteriscos, hashtags, negrito, itálico ou qualquer formatação especial. Escreva em texto simples e corrido.
Regra 10: Ao listar produtos com estoque baixo, use um formato simples e direto. Exemplo: 'Hamburgão: 2 unidades (mínimo: 20)' em vez de frases longas e repetitivas.
Regra 11: Quando o usuário perguntar o que precisa comprar ou repor no estoque, calcule a quantidade a comprar de cada produto assim: quantidade a comprar = quantidade_min - quantidade_atual. Mostre apenas o nome e a quantidade a comprar. Exemplo: 'Hamburgão: 18 unidades'.
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

def extrair_empresa_id(token: str) -> int | None:
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload.get("empresaId") or payload.get("empresa_id") or payload.get("companyId")
    except Exception:
        return None

@app.post("/api/chat")
async def chat_vendas(req: RequisicaoChat):
    try:
        hoje = date.today()
        inicio_semana = hoje - timedelta(days=hoje.weekday())
        inicio_mes = hoje.replace(day=1)

        empresa_id = extrair_empresa_id(req.token) or req.id_usuario

        contexto_roteador = f"""
        Contexto:
        - ID da empresa: {empresa_id}
        - ID do funcionário: {req.id_usuario}
        - Data de hoje: {hoje}
        - Data de ontem: {hoje - timedelta(days=1)}
        - Início da semana: {inicio_semana}
        - Início do mês: {inicio_mes}

        Pergunta do usuário: {req.pergunta}

        ATENÇÃO:
        - Se a pergunta mencionar 'semana', o endpoint DEVE incluir ?dataInicio={inicio_semana}&dataFim={hoje}.
        - Se mencionar 'mês', o endpoint DEVE incluir ?dataInicio={inicio_mes}&dataFim={hoje}.
        - Se mencionar 'ontem', o endpoint DEVE incluir ?dataInicio={hoje - timedelta(days=1)}&dataFim={hoje - timedelta(days=1)}.
        - 'Vendas da semana' ou 'o que vendemos essa semana' = /vendas/pratos-vendidos/{{empresaId}} com datas.
        - 'Quantas vendas' = /vendas/quantidade-vendas/{{empresaId}} com datas.
        - 'KPIs' ou 'indicadores' = /vendas/kpis/{{empresaId}}.
        - Nunca use /vendas/kpis para perguntas genéricas sobre vendas.
        """

        resposta_rota = model_router.generate_content(contexto_roteador)
        endpoint_escolhido = resposta_rota.text.strip()

        print(f"Endpoint escolhido: {endpoint_escolhido}")

        dados_vendas = {}

        if endpoint_escolhido != "NENHUM":
            headers = {"Authorization": f"Bearer {req.token}"}
            url_java = f"http://localhost:8080{endpoint_escolhido}"
            resposta_java = requests.get(url_java, headers=headers)
            dados_vendas = resposta_java.json()

        prompt_completo = f"""
        Contexto do usuário logado:
        - ID da empresa: {empresa_id}
        - ID do funcionário: {req.id_usuario}
        - Data de hoje: {hoje}

        Dados retornados pelo sistema:
        {dados_vendas}

        Pergunta do usuário: {req.pergunta}
        """

        resposta_ia = model_assistente.generate_content(prompt_completo)
        return {"resposta": resposta_ia.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))