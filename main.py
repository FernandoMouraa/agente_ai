""" Execução principal do projeto: triagem de mensagens e RAG. """

from triagem import triagem
from loader_rag import carregar_docs, criar_retriever
from rag import formatar_citacoes
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from models import llm_triagem

# Carrega documentos e cria retriever
docs = carregar_docs()
retriever = criar_retriever(docs)

# Prompt para RAG
prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
     "Responda SOMENTE com base no contexto fornecido. Se não houver base suficiente, responda apenas 'Não sei'."),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

# Chain para gerar respostas contextualizadas
document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

# Testes de triagem
testes_triagem = [
    "Posso reembolsar a internet?",
    "Quero mais 5 dias de trabalho remoto. Como faço?"
]

print("=== TRIAGEM ===")
for msg in testes_triagem:
    print(f"Pergunta: {msg}")
    print(f"Resposta: {triagem(msg)}\n")

# Testes RAG
testes_rag = [
    "Posso reembolsar cursos ou treinamentos da Alura?",
    "Quantas capivaras tem no Rio Pinheiros?"
]

print("=== RAG ===")
for msg in testes_rag:
    docs_relacionados = retriever.invoke(msg)
    if not docs_relacionados:
        print(f"Pergunta: {msg}\nResposta: Não sei.\n")
        continue

    answer = document_chain.invoke({"input": msg, "context": docs_relacionados}).strip()
    if answer.rstrip(".!?") == "Não sei":
        answer = "Não sei."

    citacoes = formatar_citacoes(docs_relacionados, msg)
    print(f"Pergunta: {msg}\nResposta: {answer}")
    if citacoes:
        print("Citações:")
        for c in citacoes:
            print(f" - Documento: {c['documento']}, Página: {c['pagina']}")
            print(f"   Trecho: {c['trecho']}")
    print("------------------------------------")
