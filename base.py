# Imports do LangChain e do modelo Google
from langchain_google_genai import ChatGoogleGenerativeAI
# Imports para usar .env
from dotenv import load_dotenv
import os

# Carrega o arquivo .env
load_dotenv()

#Instância do LLM principal
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,  # temperatura menor para respostas mais objetivas
    google_api_key=os.getenv("GEMINI_API_KEY")  # pega a variável do .env
)

# Teste simples do LLM
resposta = llm.invoke("Quem é você, seja criativo")
print("Resposta LLM:", resposta.content)