import hashlib
import json
import re
from pathlib import Path

from atm_agent.config.paths import DATA

KNOWLEDGE = DATA / 'knowledge'
INDEX = DATA / 'rag'
CHUNKING = {'version': '1.0.0', 'max_chars': 2600, 'overlap_chars': 350}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def split_page(text: str, max_chars: int = 2600, overlap_chars: int = 350) -> list[str]:
    """Режет по строкам с перекрытием; сохраняет колонки и не объединяет страницы редакций."""
    if max_chars <= overlap_chars or overlap_chars < 0:
        raise ValueError('Перекрытие должно быть меньше размера фрагмента.')
    lines = []
    for line in text.replace('\xa0', ' ').splitlines():
        if re.match(r'^\s*(Редакция документа с|Страница \d+ из|V\.?\s*\d{2}\.\d{2}\.\d{4})', line):
            continue
        if line.strip():
            lines.extend(line[start:start + max_chars] for start in range(0, len(line), max_chars))
    chunks, current = [], ''
    for line in lines:
        if current and len(current) + len(line) + 1 > max_chars:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ''
            current = tail if len(tail) + len(line) + 1 <= max_chars else ''
        current = f'{current}\n{line}' if current else line
    if current:
        chunks.append(current)
    return chunks


def make_chunks(documents: list[dict], pages: list[dict]) -> list[dict]:
    """Разбивает страницы на перекрывающиеся фрагменты, сохраняя редакцию, страницу и источник."""
    sources = {document['source_id']: document for document in documents}
    chunks = []
    for page in pages:
        source = sources[page['source_id']]
        text = page.get('layout_text') or page['text']
        for number, body in enumerate(split_page(text, **{k: v for k, v in CHUNKING.items() if k != 'version'})):
            prefix = (f"{source['title']}\nРедакция: {source['version']}. "
                      f"Документ: {source['document_id']}. Файл: {Path(source['path']).name}. "
                      f"Страница {page['page']}.\n")
            chunks.append({
                'chunk_id': f"{page['page_id']}:c{number:02}",
                'source_id': source['source_id'], 'document_id': source['document_id'],
                'product': source['product'], 'title': source['title'], 'version': source['version'],
                'page': page['page'], 'source_path': source['path'],
                'valid_from': source['valid_from'], 'valid_to': source['valid_to'],
                'text': body, 'index_text': prefix + body,
            })
    return chunks


def corpus_signature(knowledge: Path = KNOWLEDGE) -> str:
    return fingerprint({
        'catalog': json.loads((knowledge / 'catalog.json').read_text()),
        'documents_hash': hashlib.sha256((knowledge / 'documents.jsonl').read_bytes()).hexdigest(),
        'pages_hash': hashlib.sha256((knowledge / 'pages.jsonl').read_bytes()).hexdigest(),
        'chunking': CHUNKING,
    })
