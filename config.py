# Configurações gerais e variáveis de ambiente
from dotenv import load_dotenv
import os

# Carrega o arquivo .env
load_dotenv()

# Chave API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
