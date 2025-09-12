# Fernando AI - Assistente de Políticas Internas

Projeto em Python para triagem de mensagens de Service Desk e busca contextualizada (RAG) em PDFs de políticas internas.

## Estrutura

- `config.py`: Variáveis de ambiente e prompts fixos.
- `models.py`: Modelos Pydantic e LLMs.
- `triagem.py`: Função de triagem de mensagens.
- `loader_rag.py`: Carregamento de PDFs e criação do retriever FAISS.
- `rag.py`: Funções de RAG, formatação de citações.
- `main.py`: Execução e testes.
- `docs/`: PDFs de políticas internas (coloque aqui seus arquivos).
- `.env`: Arquivo com `GEMINI_API_KEY=<sua_chave>`.

## Requisitos

- Python >= 3.10
- Instalar dependências:
```bash
pip install -r requirements.txt

# Clone o repositório
https://github.com/FernandoMouraa/agente_ai.git


