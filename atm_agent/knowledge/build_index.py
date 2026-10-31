import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from atm_agent.config.settings import connection_options, create_embeddings
from atm_agent.knowledge.corpus import (
    CHUNKING,
    INDEX,
    KNOWLEDGE,
    corpus_signature,
    make_chunks,
    read_jsonl,
)
from atm_agent.knowledge.documents import write_jsonl
from atm_agent.retrieval.index import EmbeddingCache, LexicalIndex, LocalReranker


def build(embeddings, folder: Path = INDEX) -> dict:
    """Строит индекс из подготовленных страниц, переиспользует кэш векторов и сохраняет манифест корпуса."""
    documents = read_jsonl(KNOWLEDGE / 'documents.jsonl')
    pages = read_jsonl(KNOWLEDGE / 'pages.jsonl')
    chunks = make_chunks(documents, pages)
    folder.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(embeddings.model, folder / 'embedding_cache.sqlite3')
    texts = [item['index_text'] for item in chunks]
    missing = [text for text in dict.fromkeys(texts) if cache.get(text) is None]
    print(f'Фрагментов: {len(chunks)}; новых эмбеддингов: {len(missing)}; модель: {embeddings.model}', flush=True)
    try:
        for start in range(0, len(missing), 16):
            batch = missing[start:start + 16]
            for attempt in range(3):
                try:
                    vectors = embeddings.embed_documents(batch)
                    if len(vectors) != len(batch):
                        raise ValueError('Число эмбеддингов не совпало с числом текстов.')
                    break
                except Exception as exc:
                    status = getattr(exc, 'status_code', None)
                    if status and status not in (429, 500, 502, 503, 504):
                        raise
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            for text, vector in zip(batch, vectors):
                cache.put(text, vector)
            print(f'Новые эмбеддинги: {min(start + len(batch), len(missing))}/{len(missing)}', flush=True)
        vectors = np.asarray([cache.get(text) for text in texts], dtype=np.float32)
    finally:
        cache.close()
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError('Не удалось получить согласованный массив векторов.')
    write_jsonl(folder / 'chunks.jsonl', chunks)
    write_jsonl(folder / 'documents.jsonl', documents)
    with (folder / 'vectors.tmp').open('wb') as stream:
        np.save(stream, vectors, allow_pickle=False)
    (folder / 'vectors.tmp').replace(folder / 'vectors.npy')
    manifest = {
        'corpus_signature': corpus_signature(), 'chunking': CHUNKING,
        'embedding_model': embeddings.model, 'dimensions': vectors.shape[1],
        'embedding_usage_this_run': getattr(embeddings, 'usage', None),
        'query_prefix': embeddings.prefix_query if embeddings.use_prefix_query else '',
        'chunk_count': len(chunks), 'document_count': len(documents),
        'bm25': {'backend': 'bm25s', 'k1': 1.5, 'b': 0.75, 'lemmatizer': 'pymorphy3'},
        'fusion': {'algorithm': 'RRF', 'rank_constant': 60},
        'reranker': 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
        'reranker_revision': json.loads((folder / 'reranker.json').read_text())['revision']
        if (folder / 'reranker.json').exists() else None,
    }
    (folder / 'index.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def prepare_reranker():
    """Подготавливает локальные веса реранкера и фиксирует ревизию для воспроизводимой загрузки."""
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import model_info

    reranker = LocalReranker()
    INDEX.mkdir(parents=True, exist_ok=True)
    path = INDEX / 'reranker.json'
    metadata = json.loads(path.read_text()) if path.exists() else {
        'model': reranker.name, 'revision': model_info(reranker.name).sha,
    }
    if metadata.get('ready'):
        reranker.load(allow_download=True)
        print(json.dumps(metadata, indent=2), flush=True)
        return
    metadata['ready'] = False
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    reranker.load(allow_download=True)
    metadata['ready'] = True
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


def lexical_check():
    """Проверяет реальный корпус и локальный реранкер; не подменяет проверку гибридного поиска."""
    from atm_agent.knowledge.versions import select_sources

    documents = read_jsonl(KNOWLEDGE / 'documents.jsonl')
    chunks = make_chunks(documents, read_jsonl(KNOWLEDGE / 'pages.jsonl'))
    lexical, reranker = LexicalIndex(chunks), LocalReranker()
    query = 'Структура договора СберПрайм: какие приложения входят в договор?'
    report = {'mode': 'lexical_and_cross_encoder_only', 'dense_search_used': False, 'query': query, 'runs': []}
    for day in ('2026-03-01', '2026-09-30'):
        allowed_sources = {item['source_id'] for item in select_sources(documents, day, document_id='sberprime-offer')}
        allowed = {index for index, item in enumerate(chunks) if item['source_id'] in allowed_sources}
        run = lexical.search(query, allowed, 12)
        pool = [chunks[index] for index, _ in run]
        ranked = reranker.rank(query, pool)
        mapping = {item['chunk_id']: item for item in pool}
        top = [{'chunk_id': key, 'score': score, 'page': mapping[key]['page'],
                'text': mapping[key]['text']} for key, score in ranked[:5]]
        if not any(item['page'] == 1 for item in top):
            raise ValueError('В результатах не найдена первая страница со структурой договора.')
        report['runs'].append({'as_of': day, 'sources': sorted(allowed_sources), 'top': top})
    path = INDEX / 'lexical_check.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'report': str(path), 'runs': [{
        'as_of': run['as_of'], 'sources': run['sources'], 'pages': [item['page'] for item in run['top']],
    } for run in report['runs']]}, ensure_ascii=False, indent=2))


def main():
    """Запускает сборку индекса или выбранную диагностическую проверку по аргументам командной строки."""
    parser = argparse.ArgumentParser(description='Подготовка индекса BM25 + embeddings + Qdrant')
    parser.add_argument('--env-file')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--probe-sdk', action='store_true')
    parser.add_argument('--probe-rest', action='store_true')
    parser.add_argument('--embedding-balance', action='store_true')
    parser.add_argument('--prepare-reranker', action='store_true')
    parser.add_argument('--check-reranker', action='store_true')
    parser.add_argument('--lexical-check', action='store_true')
    parser.add_argument('--chunks-only', action='store_true')
    parser.add_argument('--embedding-model')
    args = parser.parse_args()
    if args.embedding_balance:
        from gigachat import GigaChat

        with GigaChat(**connection_options(args.env_file)) as client:
            response = client.get_balance()
            entries = [{'model': item.usage, 'tokens': item.value} for item in response.balance
                       if 'embed' in item.usage.lower()]
            print(json.dumps({'embedding_balance': entries,
                              'reported_model_names': [item.usage for item in response.balance]}, indent=2))
        return
    if args.probe_rest:
        import requests
        from gigachat import GigaChat

        options = connection_options(args.env_file)
        with GigaChat(**options) as client:
            token = client.get_token()
            if token is None:
                raise ValueError('Не получен access token.')
            response = requests.post(
                options.get('base_url', 'https://api.giga.chat/v1').rstrip('/') + '/embeddings',
                headers={'Authorization': 'Bearer ' + token.access_token, 'Accept': 'application/json'},
                json={'model': args.embedding_model or 'Embeddings',
                      'input': 'Расскажи о современных технологиях'},
                verify=options.get('ca_bundle_file') or options['verify_ssl_certs'], timeout=30,
            )
            body = response.json()
            print(json.dumps({'http_status': response.status_code, 'model': body.get('model'),
                              'vectors_received': len(body.get('data', [])),
                              'dimensions': len(body['data'][0]['embedding']) if body.get('data') else None,
                              'payment_required': response.status_code == 402}, indent=2))
            if not response.ok:
                raise SystemExit(1)
        return
    if args.probe_sdk:
        from gigachat import GigaChat

        options = connection_options(args.env_file)
        with GigaChat(base_url=options.get('base_url', 'https://api.giga.chat/v1'),
                      credentials=options['credentials'], verify_ssl_certs=options['verify_ssl_certs']) as client:
            response = client.embeddings(['Hello world!'], model=args.embedding_model or 'EmbeddingsGigaR')
            print(json.dumps({'model': response.model,
                              'dimensions': len(response.data[0].embedding),
                              'usage': [item.usage.model_dump() for item in response.data]}, indent=2))
        return
    if args.prepare_reranker:
        prepare_reranker()
        return
    if args.lexical_check:
        lexical_check()
        return
    if args.check_reranker:
        reranker = LocalReranker()
        ranked = reranker.rank('За что начисляются бонусы?', [
            {'chunk_id': 'relevant', 'index_text': 'Бонусы начисляются участнику за покупки по карте.'},
            {'chunk_id': 'unrelated', 'index_text': 'Объявляется посадка на самолёт в аэропорту.'},
        ])
        print(json.dumps({'model': reranker.name, 'ranked': ranked}, ensure_ascii=False, indent=2))
        if ranked[0][0] != 'relevant':
            raise ValueError('Контрольный пример реранкера не пройден.')
        return
    if args.chunks_only:
        chunks = make_chunks(read_jsonl(KNOWLEDGE / 'documents.jsonl'), read_jsonl(KNOWLEDGE / 'pages.jsonl'))
        inventory = {'chunking': CHUNKING, 'chunk_count': len(chunks),
                     'max_chars': max(len(item['index_text']) for item in chunks),
                     'dense_index_built': False}
        INDEX.mkdir(parents=True, exist_ok=True)
        (INDEX / 'chunk_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
        print(json.dumps(inventory, indent=2))
        return
    if args.embedding_model:
        os.environ['GIGACHAT_EMBEDDING_MODEL'] = args.embedding_model
    embeddings = create_embeddings(args.env_file)
    if args.probe:
        vector = embeddings.embed_query('Какие сервисы входят в СберПрайм?')
        print(json.dumps({'model': embeddings.model, 'dimensions': len(vector),
                          'usage': getattr(embeddings, 'usage', None)}, indent=2))
        return
    print(json.dumps(build(embeddings), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    from atm_agent.config.diagnostics import connection_error

    try:
        main()
    except Exception as exc:
        from atm_agent.retrieval.embeddings import CloudEmbeddingError

        if isinstance(exc, CloudEmbeddingError):
            print(str(exc), flush=True)
        elif getattr(exc, 'status_code', None):
            print(connection_error(exc), flush=True)
        else:
            print(f'Подготовка индекса не завершена ({type(exc).__name__}).', flush=True)
        raise SystemExit(1)
