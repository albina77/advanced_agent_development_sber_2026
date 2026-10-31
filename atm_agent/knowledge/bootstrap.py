import json

import numpy as np

from atm_agent.config.settings import create_embeddings
from atm_agent.knowledge.build_index import build, prepare_reranker
from atm_agent.knowledge.corpus import INDEX, KNOWLEDGE, corpus_signature, read_jsonl
from atm_agent.knowledge.documents import REPOSITORY, check_document, prepare
from atm_agent.knowledge.versions import resolve_intervals
from atm_agent.retrieval.index import LocalReranker


def prepare_sources():
    """Проверяет исходные PDF и обновляет извлечённый корпус при изменении каталога."""
    catalog_path = KNOWLEDGE / 'catalog.json'
    catalog = json.loads(catalog_path.read_text())
    documents = resolve_intervals(catalog['documents'], catalog['temporal_policy'])
    for document in documents:
        check_document(document, REPOSITORY)
    try:
        saved = read_jsonl(KNOWLEDGE / 'documents.jsonl')
        ready = (KNOWLEDGE / 'pages.jsonl').is_file() and len(saved) == len(documents)
        ready = ready and all(all(old.get(key) == value for key, value in new.items())
                              for old, new in zip(saved, documents))
    except (OSError, ValueError):
        ready = False
    if not ready:
        print('Подготовка текста PDF и редакций…', flush=True)
        prepare(catalog_path, KNOWLEDGE)


def index_ready(embeddings):
    """Проверяет комплект индекса, отпечаток корпуса и совместимость модели без API-запросов."""
    try:
        manifest = json.loads((INDEX / 'index.json').read_text())
        prefix = embeddings.prefix_query if embeddings.use_prefix_query else ''
        if (manifest['corpus_signature'] != corpus_signature()
                or manifest['embedding_model'] != embeddings.model
                or manifest.get('query_prefix') != prefix):
            return False
        chunks = read_jsonl(INDEX / 'chunks.jsonl')
        documents = read_jsonl(INDEX / 'documents.jsonl')
        vectors = np.load(INDEX / 'vectors.npy', allow_pickle=False, mmap_mode='r')
        return (vectors.shape == (len(chunks), manifest['dimensions'])
                and len(chunks) == manifest['chunk_count']
                and len(documents) == manifest['document_count']
                and bool(np.isfinite(vectors).all()))
    except (OSError, ValueError, KeyError):
        return False


def ensure_ready(env_file=None):
    """Готовит недостающие ресурсы и возвращает эмбеддер с загруженным локальным реранкером."""
    embeddings = create_embeddings(env_file)
    prepare_sources()
    reranker = LocalReranker()
    try:
        reranker.load()
    except (OSError, ValueError):
        print('Подготовка локального реранкера: первый запуск требует загрузки модели.', flush=True)
        prepare_reranker()
        reranker.load()
    if not index_ready(embeddings):
        print('Подготовка индекса: новые эмбеддинги расходуют оплаченные токены провайдера.', flush=True)
        build(embeddings)
    return embeddings, reranker
