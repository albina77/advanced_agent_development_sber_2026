import argparse
import json
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np

from atm_agent.config.paths import DATA
from atm_agent.memory.client import MemoryRecord
from atm_agent.memory.store import SqliteStore, parse_timestamp


def record_text(value):
    record = MemoryRecord.model_validate(value)
    if record.notes:
        choice = ', '.join(f'{count} купюр по {denomination} рублей' for denomination, count in record.notes)
    elif record.denomination:
        choice = f'купюры по {record.denomination} рублей'
    else:
        choice = 'выбор купюр банкоматом'
    if record.kind == 'cash_preference':
        return f'Явное предпочтение клиента для будущих снятий наличных: {choice}.'
    return f'Успешная выдача наличных: {record.amount} рублей, {choice}.'


def claim_batch(store, model, limit):
    """Короткая аренда заданий защищает от двух одновременных обработчиков."""
    now, token = store.clock(), uuid4().hex
    with store.lock, store.connection:
        store.connection.execute('BEGIN IMMEDIATE')
        rows = store.connection.execute('''SELECT m.namespace, m.key, m.revision, m.value
            FROM memory m LEFT JOIN memory_vectors v
            ON m.namespace=v.namespace AND m.key=v.key AND m.revision=v.revision AND v.model=?
            WHERE v.key IS NULL''', (model,)).fetchall()
        for namespace, key, revision, _ in rows:
            if store._personal(tuple(json.loads(namespace))):
                store.connection.execute('''INSERT OR IGNORE INTO memory_jobs
                    (namespace, key, revision, retry_at) VALUES (?, ?, ?, ?)''',
                    (namespace, key, revision, now.isoformat()))
        jobs = store.connection.execute('''SELECT j.namespace, j.key, j.revision, m.value, j.attempts
            FROM memory_jobs j JOIN memory m
            ON j.namespace=m.namespace AND j.key=m.key AND j.revision=m.revision
            WHERE j.attempts<3 AND j.retry_at<=? AND (j.lease_until='' OR j.lease_until<=?)
            ORDER BY j.retry_at, j.namespace, j.key LIMIT ?''', (now.isoformat(), now.isoformat(), limit)).fetchall()
        selected = []
        for namespace, key, revision, value, attempts in jobs:
            record = json.loads(value)
            if not parse_timestamp(record['created_at']) <= now < parse_timestamp(record['expires_at']):
                continue
            store.connection.execute('''UPDATE memory_jobs SET lease_token=?, lease_until=?, attempts=attempts+1
                WHERE namespace=? AND key=?''',
                (token, (now + timedelta(minutes=5)).isoformat(), namespace, key))
            selected.append({'namespace': namespace, 'key': key, 'revision': revision,
                             'value': record, 'attempts': attempts + 1, 'lease_token': token})
    return selected


def process_batch(store, embeddings, jobs):
    """Вычисляет векторы вне транзакции и сохраняет только для актуальных версий с действующей арендой."""
    if not jobs:
        return {'indexed': 0, 'discarded': 0, 'failed': 0}
    try:
        vectors = embeddings.embed_documents([record_text(job['value']) for job in jobs])
        if len(vectors) != len(jobs):
            raise ValueError('Число векторов памяти не совпало с числом записей.')
        arrays = [np.asarray(vector, dtype=np.float32) for vector in vectors]
        if any(array.ndim != 1 or not len(array) or not np.isfinite(array).all() or not np.linalg.norm(array)
               for array in arrays) or len({len(array) for array in arrays}) != 1:
            raise ValueError('Некорректные векторы памяти.')
    except Exception as exc:
        with store.lock, store.connection:
            for job in jobs:
                store.connection.execute('''UPDATE memory_jobs SET error_type=?, retry_at=?, lease_token='', lease_until=''
                    WHERE namespace=? AND key=? AND revision=? AND lease_token=?''',
                    (type(exc).__name__, (store.clock() + timedelta(seconds=2 ** job['attempts'])).isoformat(),
                     job['namespace'], job['key'], job['revision'], job['lease_token']))
                store._audit(tuple(json.loads(job['namespace'])), job['key'], 'index_failed', type(exc).__name__)
        return {'indexed': 0, 'discarded': 0, 'failed': len(jobs)}
    indexed = 0
    with store.lock, store.connection:
        store.connection.execute('BEGIN IMMEDIATE')
        for job, vector in zip(jobs, arrays):
            current = store.connection.execute('''SELECT m.value FROM memory m JOIN memory_jobs j
                ON m.namespace=j.namespace AND m.key=j.key AND m.revision=j.revision
                WHERE m.namespace=? AND m.key=? AND m.revision=? AND j.lease_token=?''',
                (job['namespace'], job['key'], job['revision'], job['lease_token'])).fetchone()
            if not current or parse_timestamp(json.loads(current[0])['expires_at']) <= store.clock():
                continue
            store.connection.execute('INSERT OR REPLACE INTO memory_vectors VALUES (?, ?, ?, ?, ?)',
                (job['namespace'], job['key'], job['revision'], embeddings.model, vector.tobytes()))
            store.connection.execute('DELETE FROM memory_jobs WHERE namespace=? AND key=? AND revision=?',
                                    (job['namespace'], job['key'], job['revision']))
            store._audit(tuple(json.loads(job['namespace'])), job['key'], 'index', 'background_worker')
            indexed += 1
    return {'indexed': indexed, 'discarded': len(jobs) - indexed, 'failed': 0}


def run_once(store, embeddings, limit=100):
    """Один ограниченный проход: очистка, пакетные эмбеддинги, условная запись результата."""
    if limit <= 0:
        raise ValueError('Лимит обработки должен быть положительным.')
    report = {'expired': store.purge_expired(), 'artifacts_removed': store.cleanup_artifacts(),
              'indexed': 0, 'discarded': 0, 'failed': 0}
    for start in range(0, limit, 16):
        jobs = claim_batch(store, embeddings.model, min(16, limit - start))
        if not jobs:
            break
        result = process_batch(store, embeddings, jobs)
        for key, value in result.items():
            report[key] += value
        if result['failed']:
            break
    report['embedding_usage'] = dict(getattr(embeddings, 'usage', None) or {})
    with store.lock:
        report['pending_jobs'] = store.connection.execute('SELECT COUNT(*) FROM memory_jobs').fetchone()[0]
        report['exhausted_jobs'] = store.connection.execute('SELECT COUNT(*) FROM memory_jobs WHERE attempts>=3').fetchone()[0]
    return report


def main():
    """Запускает один проход обработки памяти или выводит аудит без обращения к эмбеддеру."""
    import os

    from atm_agent.config.settings import create_embeddings, load_settings

    parser = argparse.ArgumentParser(description='Отдельный обработчик памяти: эмбеддинги и очистка сроков хранения')
    parser.add_argument('--env-file')
    parser.add_argument('--memory-db', type=Path, default=DATA / 'memory/store.sqlite3')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--audit-client', help='Показать служебный аудит выбранного учебного клиента')
    args = parser.parse_args()
    load_settings(args.env_file)
    embeddings = None if args.audit_client else create_embeddings(args.env_file)
    store = SqliteStore(args.memory_db, embeddings=embeddings,
                        retention_days=int(os.getenv('MEMORY_SERVICE_RETENTION_DAYS', '365')))
    try:
        report = {'audit': store.audit(args.audit_client), 'receipts': store.receipts(args.audit_client)} \
            if args.audit_client else run_once(store, embeddings, args.limit)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        store.close()
        if hasattr(embeddings, 'close'):
            embeddings.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Обработчик памяти завершён с ошибкой ({type(exc).__name__}).')
        raise SystemExit(1)
