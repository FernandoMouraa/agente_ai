""" Funções de RAG: busca contextualizada, formatação de citações e extração de trechos. """

import re, pathlib
from typing import List, Dict

def _clean_text(s: str) -> str:
    """Remove espaços extras e quebras de linha do texto."""
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    """Retorna trecho do texto em torno das palavras da query."""
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1:
            break
    if pos == -1:
        pos = 0
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    """
    Formata citações dos documentos encontrados:
    - documento
    - página
    - trecho relevante
    """
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source","")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({
            "documento": src,
            "pagina": page,
            "trecho": extrair_trecho(d.page_content, query)
        })
    return cites[:3]
