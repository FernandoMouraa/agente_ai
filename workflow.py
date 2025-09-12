from typing import TypedDict, Optional, List, Dict
from triagem import triagem
from rag import perguntar_politica_RAG
from langgraph.graph import StateGraph, START, END
from loader_rag import carregar_docs, gerar_chunks, criar_vectorstore
from models import llm_triagem
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import time

# ------------------- Configuração de API e cache -------------------

cache_triagem: Dict[str, dict] = {}
cache_rag: Dict[str, dict] = {}
ultima_chamada = 0
DELAY = 6  # segundos entre chamadas para free tier

def respeitar_quota():
    global ultima_chamada
    agora = time.time()
    diff = agora - ultima_chamada
    if diff < DELAY:
        time.sleep(DELAY - diff)
    ultima_chamada = time.time()

# ------------------- Inicialização RAG -------------------

docs = carregar_docs("docs/")
chunks = gerar_chunks(docs)
retriever = criar_vectorstore(chunks)

prompt_rag = ChatPromptTemplate.from_messages([
    ("system", "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
               "Responda SOMENTE com base no contexto fornecido. "
               "Se não houver base suficiente, responda apenas 'Não sei'."),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])
document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

# ------------------- Tipagem do estado -------------------

class AgentState(TypedDict, total=False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str

# ------------------- Função de validação de contexto -------------------

def validar_contexto(pergunta: str) -> bool:
    palavras_chave_contexto = [
        "reembolso", "home office", "curso", "certificação",
        "treinamento", "política", "acesso", "chamado", "aprovação"
    ]
    pergunta_lower = pergunta.lower()
    return any(p in pergunta_lower for p in palavras_chave_contexto)

# ------------------- Nós do Workflow -------------------

def node_triagem(state: AgentState) -> AgentState:
    respeitar_quota()

    pergunta = state["pergunta"]

    if not validar_contexto(pergunta):
        return {
            "resposta": "Não sei responder, fora do contexto das políticas internas.",
            "citacoes": [],
            "acao_final": "FORA_CONTEXTO"
        }

    if pergunta in cache_triagem:
        tri = cache_triagem[pergunta]
    else:
        print("Executando nó de triagem...")
        tri = triagem(pergunta)
        cache_triagem[pergunta] = tri

    return {"triagem": tri}

def node_auto_resolver(state: AgentState) -> AgentState:
    respeitar_quota()

    pergunta = state["pergunta"]
    if pergunta in cache_rag:
        resposta_rag = cache_rag[pergunta]
    else:
        print("Executando nó de auto_resolver...")
        resposta_rag = perguntar_politica_RAG(pergunta, retriever, document_chain)
        cache_rag[pergunta] = resposta_rag

    update: AgentState = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes", []),
        "rag_sucesso": resposta_rag["contexto_encontrado"],
    }

    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"

    return update

def node_pedir_info(state: AgentState) -> AgentState:
    print("Executando nó de pedir_info...")
    faltantes = state["triagem"].get("campos_faltantes", [])
    detalhe = ",".join(faltantes) if faltantes else "Tema e contexto específico"

    return {
        "resposta": f"Para avançar, preciso que detalhe: {detalhe}",
        "citacoes": [],
        "acao_final": "PEDIR_INFO"
    }

def node_abrir_chamado(state: AgentState) -> AgentState:
    print("Executando nó de abrir_chamado...")
    tri = state["triagem"]

    return {
        "resposta": f"Abrindo chamado com urgência {tri['urgencia']}. Descrição: {state['pergunta'][:140]}",
        "citacoes": [],
        "acao_final": "ABRIR_CHAMADO"
    }

# ------------------- Decisões -------------------

KEYWORDS_ABRIR_TICKET = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]

def decidir_pos_triagem(state: AgentState) -> str:
    print("Decidindo após a triagem...")
    if state.get("acao_final") == "FORA_CONTEXTO":
        return "fora"
    decisao = state["triagem"]["decisao"]

    if decisao == "AUTO_RESOLVER": return "auto"
    if decisao == "PEDIR_INFO": return "info"
    if decisao == "ABRIR_CHAMADO": return "chamado"

def decidir_pos_auto_resolver(state: AgentState) -> str:
    print("Decidindo após o auto_resolver...")

    if state.get("rag_sucesso"):
        print("Rag com sucesso, finalizando o fluxo.")
        return "ok"

    state_da_pergunta = (state["pergunta"] or "").lower()
    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
        print("Rag falhou, mas foram encontradas keywords de abertura de ticket. Abrindo...")
        return "chamado"

    print("Rag falhou, sem keywords, vou pedir mais informações...")
    return "info"

# ------------------- Montagem do Workflow -------------------

workflow = StateGraph(AgentState)

workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("abrir_chamado", node_abrir_chamado)

workflow.add_edge(START, "triagem")
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info",
    "chamado": "abrir_chamado",
    "fora": END
})

workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "info": "pedir_info",
    "chamado": "abrir_chamado",
    "ok": END
})

workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_chamado", END)

grafo = workflow.compile()
