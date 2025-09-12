# Arquivo principal para rodar triagem e RAG
from triagem import triagem
from loader_rag import carregar_docs, gerar_chunks, criar_vectorstore
from rag import perguntar_politica_RAG
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from triagem import llm_triagem

# ----------------------------
# Triagem - exemplos
# ----------------------------
testes_triagem = ["Posso reembolsar a internet?",
                  "Quero mais 5 dias de trabalho remoto. Como faço?",
                  "Posso reembolsar cursos ou treinamentos da Alura?",
                  "Quantas capivaras tem no Rio Pinheiros?"]

for msg in testes_triagem:
    print(f"Pergunta: {msg}\n -> Resposta: {triagem(msg)}\n")

# ----------------------------
# RAG - inicialização
# ----------------------------
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

# ----------------------------
# RAG - testes
# ----------------------------
testes_rag = ["Posso reembolsar a internet?",
              "Quero mais 5 dias de trabalho remoto. Como faço?",
              "Posso reembolsar cursos ou treinamentos da Alura?",
              "Quantas capivaras tem no Rio Pinheiros?"]

for msg in testes_rag:
    resposta = perguntar_politica_RAG(msg, retriever, document_chain)
    print(f"PERGUNTA: {msg}")
    print(f"RESPOSTA: {resposta['answer']}")
    if resposta['contexto_encontrado']:
        print("CITAÇÕES:")
        for c in resposta['citacoes']:
            print(f" - Documento: {c['documento']}, Página: {c['pagina']}")
            print(f"   Trecho: {c['trecho']}")
        print("------------------------------------")
