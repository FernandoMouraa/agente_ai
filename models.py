""" Modelos de dados Pydantic e inicialização do LLM. """

from pydantic import BaseModel, Field
from typing import Literal, List
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY

# Classe Pydantic para validar saída da triagem
class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)

# Instância do LLM Gemini para triagem
llm_triagem = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)

# Chain estruturada para gerar saída validada
triagem_chain = llm_triagem.with_structured_output(TriagemOut)
