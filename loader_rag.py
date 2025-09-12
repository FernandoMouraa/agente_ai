""" Carregamento de documentos PDF, divisão em chunks e criação do retriever FAISS. """

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import GEMINI_API_KEY

def carregar_docs(path="docs/"):
    """
    Carrega todos os PDFs da pasta especificada.
    Retorna uma lista de documentos LangChain.
    """
    docs = []
    for n in Path(path).glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(n))
            docs.extend(loader.load())
            print(f"Carregado com sucesso arquivo {n.name}")
        except Exception as e:
            print(f"Erro ao carregar {n.name}: {e}")
    print(f"Total de documentos carregados: {len(docs)}")
    return docs

def criar_retriever(docs):
    """
    Cria embeddings e um retriever FAISS para busca de documentos.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold":0.3, "k":4}
    )
