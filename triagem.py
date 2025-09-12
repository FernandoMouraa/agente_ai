""" Função de triagem de mensagens usando LLM. """

from models import triagem_chain
from config import TRIAGEM_PROMPT
from langchain.schema import SystemMessage, HumanMessage

def triagem(mensagem: str) -> dict:
    """
    Recebe uma mensagem de usuário e retorna JSON estruturado:
    - decisao: AUTO_RESOLVER | PEDIR_INFO | ABRIR_CHAMADO
    - urgencia: BAIXA | MEDIA | ALTA
    - campos_faltantes: lista de campos que faltam
    """
    saida = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=mensagem)
    ])
    return saida.model_dump()
