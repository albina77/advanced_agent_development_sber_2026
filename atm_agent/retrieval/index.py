import json
import math
import os
import re
import sqlite3
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import bm25s
import numpy as np
import pymorphy3
from qdrant_client import QdrantClient, models

from atm_agent.knowledge.corpus import INDEX, corpus_signature, fingerprint, read_jsonl
from atm_agent.knowledge.versions import parse_date, select_sources

STOPWORDS = set('и в во на а но или по к из за от для при у о об это как какой сколько ли я вы мне'.split())


class EmbeddingCache:
    """Кэш по модели и содержимому: переживает сбои сборки и повторные запросы."""

    def __init__(self, model: str, path: Path = INDEX / 'embedding_cache.sqlite3'):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.connection = sqlite3.connect(path)
        self.connection.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB)')

    def key(self, text):
        return fingerprint([self.model, text])

    def get(self, text: str) -> list[float] | None:
        row = self.connection.execute('SELECT vector FROM vectors WHERE key = ?', (self.key(text),)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32).tolist() if row else None

    def put(self, text: str, vector: list[float]) -> None:
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or not np.linalg.norm(array):
            raise ValueError('Некорректный вектор от embedding-модели.')
        self.connection.execute('INSERT OR REPLACE INTO vectors VALUES (?, ?)', (self.key(text), array.tobytes()))
        self.connection.commit()

    def close(self):
        self.connection.close()


class Lemmatizer:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    @lru_cache(maxsize=50000)
    def lemma(self, word):
        return self.morph.parse(word)[0].normal_form if re.search('[а-яё]', word) else word

    def normalize(self, text):
        words = re.findall(r'[а-яёa-z0-9]+', text.lower())
        return ' '.join(self.lemma(word) for word in words if word not in STOPWORDS)


class LexicalIndex:
    """Лексический индекс BM25 по лемматизированному тексту фрагментов."""
    def __init__(self, chunks: list[dict]):
        self.count = len(chunks)
        self.lemmatizer = Lemmatizer()
        normalized = [self.lemmatizer.normalize(item['index_text']) for item in chunks]
        tokens = bm25s.tokenize(normalized, stopwords=None, show_progress=False, token_pattern=r'(?u)\b\w+\b')
        self.bm25 = bm25s.BM25(k1=1.5, b=0.75)
        self.bm25.index(tokens, show_progress=False)

    def search(self, query: str, allowed: set[int], top_k: int) -> list[tuple[int, float]]:
        """Фильтрует редакции до поиска, объединяет BM25 и векторы через RRF, затем применяет реранкер."""
        tokens = bm25s.tokenize([self.lemmatizer.normalize(query)], stopwords=None,
                               show_progress=False, token_pattern=r'(?u)\b\w+\b')
        indices, scores = self.bm25.retrieve(tokens, k=self.count, show_progress=False)
        return [(int(index), float(score)) for index, score in zip(indices[0], scores[0])
                if int(index) in allowed and score > 0][:top_k]


def rrf(runs: list[list[tuple[int, float]]], rank_constant: int = 60) -> list[tuple[int, float]]:
    scores = defaultdict(float)
    for run in runs:
        for rank, (index, _) in enumerate(run, 1):
            scores[index] += 1 / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class LocalReranker:
    """Локальный cross-encoder для переупорядочивания найденных кандидатов."""
    def __init__(self, name='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'):
        self.name = name
        self.model = None

    def load(self, allow_download=False):
        """Загружает закреплённую ревизию модели; сетевое скачивание требует явного разрешения аргументом."""
        if self.model is None:
            os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
            from sentence_transformers import CrossEncoder

            manifest = INDEX / 'reranker.json'
            metadata = json.loads(manifest.read_text()) if manifest.exists() else {}
            if metadata.get('model') != self.name or not metadata.get('revision'):
                raise ValueError('Сначала подготовьте реранкер через python -m atm_agent.knowledge.build_index --prepare-reranker.')
            if not metadata.get('ready') and not allow_download:
                raise ValueError('Загрузка реранкера не завершена.')
            revision = metadata['revision']
            self.model = CrossEncoder(self.name, revision=revision, device='cpu', max_length=512,
                                      cache_folder=str(INDEX / 'models'), local_files_only=not allow_download)

    def rank(self, query: str, candidates: list[dict]) -> list[tuple[str, float]]:
        if not candidates:
            return []
        self.load()
        scores = self.model.predict([(query, item['index_text']) for item in candidates],
                                    batch_size=8, show_progress_bar=False)
        ranked = [(item['chunk_id'], float(score)) for item, score in zip(candidates, scores)]
        if len(ranked) != len(candidates) or any(not math.isfinite(score) for _, score in ranked):
            raise ValueError('Некорректные оценки реранкера.')
        return sorted(ranked, key=lambda item: (-item[1], item[0]))


class HybridIndex:
    """BM25 и Qdrant с одним набором допустимых редакций до ранжирования."""

    def __init__(self, embeddings, reranker=None, folder: Path = INDEX, validate=True):
        """Загружает корпус и векторы, проверяет манифест и создаёт локальные поисковые индексы."""
        manifest = json.loads((folder / 'index.json').read_text())
        self.manifest = manifest
        if validate and manifest['corpus_signature'] != corpus_signature():
            raise ValueError('Корпус изменился. Пересоберите индекс: python -m atm_agent.knowledge.build_index.')
        if manifest['embedding_model'] != embeddings.model:
            raise ValueError('Индекс построен другой embedding-моделью.')
        prefix = getattr(embeddings, 'prefix_query', '') if getattr(embeddings, 'use_prefix_query', False) else ''
        if manifest.get('query_prefix', prefix) != prefix:
            raise ValueError('Изменилась инструкция для эмбеддингов запросов. Пересоберите индекс.')
        self.chunks = read_jsonl(folder / 'chunks.jsonl')
        self.documents = read_jsonl(folder / 'documents.jsonl')
        if validate:
            from atm_agent.knowledge.documents import REPOSITORY, check_document

            for document in self.documents:
                check_document(document, REPOSITORY)
        vectors = np.load(folder / 'vectors.npy', allow_pickle=False)
        if vectors.ndim != 2 or len(vectors) != len(self.chunks) or not np.isfinite(vectors).all():
            raise ValueError('Векторы не соответствуют фрагментам индекса.')
        self.embeddings = embeddings
        self.cache = EmbeddingCache(embeddings.model, folder / 'embedding_cache.sqlite3')
        self.reranker = reranker or LocalReranker()
        self.lexical = LexicalIndex(self.chunks)
        self.qdrant = QdrantClient(location=':memory:')
        self.qdrant.create_collection('knowledge', vectors_config=models.VectorParams(
            size=vectors.shape[1], distance=models.Distance.COSINE))
        self.qdrant.upsert('knowledge', points=[models.PointStruct(
            id=index, vector=vector.tolist(), payload={'source_id': item['source_id']},
        ) for index, (item, vector) in enumerate(zip(self.chunks, vectors))])

    def close(self):
        self.qdrant.close()
        self.cache.close()

    def search(self, query: str, as_of: str, product='', document_id='', source_id='', candidates=24, top_k=5,
               include_undated=False) -> dict:
        """Фильтрует редакции до поиска, объединяет BM25 и векторы через RRF, затем применяет реранкер."""
        parse_date(as_of)
        sources = select_sources(self.documents, as_of, document_id or None, product or None)
        if include_undated:
            sources += [item for item in self.documents if not item.get('valid_from')
                        and (not product or item['product'] == product)
                        and (not document_id or item['document_id'] == document_id)]
        if source_id:
            sources = [item for item in sources if item['source_id'] == source_id]
        allowed = {item['source_id'] for item in sources}
        trace = {'query': query, 'as_of': as_of, 'product': product,
                 'document_id': document_id, 'source_id': source_id,
                 'allowed_sources': sorted(allowed),
                 'undated_sources': [item['source_id'] for item in sources if not item.get('valid_from')],
                 'lexical': [], 'dense': [],
                 'rrf': [], 'reranked': [], 'selected': [], 'timings_ms': {}}
        keep = {index for index, item in enumerate(self.chunks) if item['source_id'] in allowed}
        if not keep or not query.strip():
            return {'chunks': [], 'trace': trace}
        start = time.perf_counter()
        lexical = self.lexical.search(query, keep, candidates)
        trace['timings_ms']['lexical'] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        prefix = getattr(self.embeddings, 'prefix_query', '') if getattr(self.embeddings, 'use_prefix_query', False) else ''
        key = prefix + query
        vector = self.cache.get(key)
        trace['query_embedding_cached'] = vector is not None
        if vector is None:
            vector = self.embeddings.embed_query(query)
            self.cache.put(key, vector)
        hits = self.qdrant.query_points(
            'knowledge', query=vector, limit=candidates,
            query_filter=models.Filter(must=[models.FieldCondition(
                key='source_id', match=models.MatchAny(any=sorted(allowed)),
            )]),
        ).points
        dense = [(int(hit.id), float(hit.score)) for hit in hits]
        trace['timings_ms']['dense'] = (time.perf_counter() - start) * 1000
        fused = rrf([lexical, dense])[:12]
        pool = [self.chunks[index] for index, _ in fused]
        start = time.perf_counter()
        reranked = self.reranker.rank(query, pool)
        trace['timings_ms']['rerank'] = (time.perf_counter() - start) * 1000
        expected = {item['chunk_id'] for item in pool}
        if len(reranked) != len(expected) or {key for key, _ in reranked} != expected:
            raise ValueError('Реранкер вернул чужие или повторяющиеся фрагменты.')
        mapping = {item['chunk_id']: item for item in pool}
        selected = [mapping[key] for key, _ in reranked[:top_k]]
        for name, run in [('lexical', lexical), ('dense', dense), ('rrf', fused)]:
            trace[name] = [{'chunk_id': self.chunks[index]['chunk_id'], 'score': score} for index, score in run]
        trace['reranked'] = [{'chunk_id': key, 'score': score} for key, score in reranked]
        trace['selected'] = [item['chunk_id'] for item in selected]
        return {'chunks': selected, 'trace': trace}
