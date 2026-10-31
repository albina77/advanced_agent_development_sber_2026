import asyncio
import hashlib
import hmac
import json
import math
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

import numpy as np
from langgraph.store.base import BaseStore, GetOp, Item, PutOp, SearchItem, SearchOp

from atm_agent.config.paths import DATA


def parse_timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Время памяти должно содержать часовой пояс.')
    return parsed


class SqliteStore(BaseStore):
    """Store с атомарными изменениями, аудитом и очередью индексации.

    Сетевые вызовы выполняются вне транзакций. Вектор принимается только для
    существующей версии записи: удаление или обновление отменяет старую работу.
    """

    def __init__(self, path: Path, embeddings=None, retention_days=365, clock=None, artifact_root=None,
                 min_similarity=0.10):
        """Открывает SQLite и подготавливает схему, аудит, очередь и настройки служебного хранения."""
        super().__init__()
        if type(retention_days) is not int or retention_days <= 0:
            raise ValueError('Срок служебного хранения должен быть положительным.')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.embeddings, self.retention_days = embeddings, retention_days
        if not 0 <= min_similarity <= 1:
            raise ValueError('Порог близости должен быть от 0 до 1.')
        self.min_similarity = min_similarity
        self.artifact_root = (artifact_root or DATA / 'rag/runs').resolve()
        self.connection = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.connection.execute('PRAGMA secure_delete=ON')
        self.lock = RLock()
        with self.connection:
            self.connection.executescript('''
                CREATE TABLE IF NOT EXISTS memory (
                    namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, key));
                CREATE TABLE IF NOT EXISTS memory_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_audit (
                    id INTEGER PRIMARY KEY, client_token TEXT NOT NULL, operation TEXT NOT NULL,
                    role TEXT NOT NULL, key_token TEXT NOT NULL, occurred_at TEXT NOT NULL,
                    retain_until TEXT NOT NULL, reason TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_jobs (
                    namespace TEXT NOT NULL, key TEXT NOT NULL, revision TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0, retry_at TEXT NOT NULL, error_type TEXT,
                    PRIMARY KEY(namespace, key));
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    namespace TEXT NOT NULL, key TEXT NOT NULL, revision TEXT NOT NULL,
                    model TEXT NOT NULL, vector BLOB NOT NULL,
                    PRIMARY KEY(namespace, key));
                CREATE TABLE IF NOT EXISTS memory_receipts (
                    client_token TEXT NOT NULL, request_id TEXT NOT NULL, action TEXT NOT NULL,
                    status TEXT NOT NULL, completed_at TEXT NOT NULL, retain_until TEXT NOT NULL,
                    PRIMARY KEY(client_token, request_id));
                CREATE TABLE IF NOT EXISTS memory_artifacts (
                    client_token TEXT NOT NULL, path TEXT NOT NULL, delete_pending INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(client_token, path));
            ''')
            self.connection.execute('INSERT OR IGNORE INTO memory_settings VALUES (?, ?)',
                                    ('token_secret', secrets.token_hex(32)))
            self.secret = bytes.fromhex(self.connection.execute(
                "SELECT value FROM memory_settings WHERE key='token_secret'").fetchone()[0])
            if 'revision' not in [row[1] for row in self.connection.execute('PRAGMA table_info(memory)')]:
                self.connection.execute("ALTER TABLE memory ADD COLUMN revision TEXT NOT NULL DEFAULT ''")
            job_columns = [row[1] for row in self.connection.execute('PRAGMA table_info(memory_jobs)')]
            for name in ('lease_token', 'lease_until'):
                if name not in job_columns:
                    self.connection.execute(f"ALTER TABLE memory_jobs ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
            if 'expires_at' not in [row[1] for row in self.connection.execute('PRAGMA table_info(memory_artifacts)')]:
                self.connection.execute("ALTER TABLE memory_artifacts ADD COLUMN expires_at TEXT NOT NULL DEFAULT ''")
            self.connection.execute("UPDATE memory_artifacts SET expires_at=? WHERE expires_at='' AND delete_pending=0",
                                    ((self.clock() + timedelta(days=30)).isoformat(),))
            for namespace, key in self.connection.execute("SELECT namespace, key FROM memory WHERE revision='' ").fetchall():
                revision = uuid4().hex
                self.connection.execute('UPDATE memory SET revision=? WHERE namespace=? AND key=?',
                                        (revision, namespace, key))
                self._enqueue(namespace, key, revision)
                ns = tuple(json.loads(namespace))
                value = json.loads(self.connection.execute(
                    'SELECT value FROM memory WHERE namespace=? AND key=?', (namespace, key)).fetchone()[0])
                if self._personal(ns) and value.get('kind') == 'withdrawal_episode':
                    created = parse_timestamp(value['created_at'])
                    self.connection.execute('INSERT OR IGNORE INTO memory_receipts VALUES (?, ?, ?, ?, ?, ?)',
                        (self.client_token(ns[1]), key, 'withdraw', 'completed', created.isoformat(),
                         (created + timedelta(days=self.retention_days)).isoformat()))

    def client_token(self, client_id):
        return hmac.new(self.secret, str(client_id).encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def _personal(namespace):
        return len(namespace) >= 3 and namespace[0] == 'clients'

    def _audit(self, namespace, key, operation, reason='dialogue'):
        if not self._personal(namespace):
            return
        now = self.clock()
        self.connection.execute('INSERT INTO memory_audit VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)', (
            self.client_token(namespace[1]), operation, namespace[2], self.client_token('key:' + key),
            now.isoformat(), (now + timedelta(days=self.retention_days)).isoformat(), reason,
        ))

    def _enqueue(self, namespace, key, revision):
        if not self._personal(tuple(json.loads(namespace))):
            return
        self.connection.execute('DELETE FROM memory_vectors WHERE namespace=? AND key=?', (namespace, key))
        self.connection.execute('''INSERT OR REPLACE INTO memory_jobs
            (namespace, key, revision, attempts, retry_at, error_type) VALUES (?, ?, ?, 0, ?, NULL)''',
                                (namespace, key, revision, self.clock().isoformat()))

    def batch(self, ops):
        ops = list(ops)
        if any(isinstance(op, SearchOp) and op.query is not None for op in ops):
            if len(ops) != 1:
                raise NotImplementedError('Семантический поиск выполняется отдельной операцией.')
            return [self._semantic(ops[0])]
        with self.lock, self.connection:
            return [self._apply(op) for op in ops]

    async def abatch(self, ops):
        return await asyncio.to_thread(self.batch, list(ops))

    def _apply(self, op, reason='dialogue'):
        """Выполняет операцию Store внутри транзакции вызывающего кода, обновляя аудит и очередь."""
        if isinstance(op, GetOp):
            row = self.connection.execute(
                'SELECT namespace, key, value, created_at, updated_at FROM memory WHERE namespace=? AND key=?',
                (json.dumps(op.namespace), op.key),
            ).fetchone()
            if row:
                self._audit(op.namespace, op.key, 'read', reason)
            return self._item(row) if row else None
        if isinstance(op, PutOp):
            if op.ttl is not None:
                raise NotImplementedError('Срок действия задаётся в expires_at записи ClientMemory.')
            namespace = json.dumps(op.namespace)
            exists = self.connection.execute('SELECT 1 FROM memory WHERE namespace=? AND key=?',
                                             (namespace, op.key)).fetchone()
            if op.value is None:
                self.connection.execute('DELETE FROM memory WHERE namespace=? AND key=?', (namespace, op.key))
                for table in ('memory_vectors', 'memory_jobs'):
                    self.connection.execute(f'DELETE FROM {table} WHERE namespace=? AND key=?', (namespace, op.key))
                if exists:
                    self._audit(op.namespace, op.key, 'delete', reason)
            else:
                now, revision = self.clock().isoformat(), uuid4().hex
                self.connection.execute('''INSERT INTO memory VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, key) DO UPDATE SET value=excluded.value,
                    updated_at=excluded.updated_at, revision=excluded.revision''',
                    (namespace, op.key, json.dumps(dict(op.value), ensure_ascii=False, allow_nan=False), now, now, revision))
                self._enqueue(namespace, op.key, revision)
                self._audit(op.namespace, op.key, 'update' if exists else 'create', reason)
            return None
        if isinstance(op, SearchOp):
            if any(isinstance(value, dict) for value in (op.filter or {}).values()):
                raise NotImplementedError('Поддерживаются только фильтры по равенству полей.')
            rows = self.connection.execute(
                'SELECT namespace, key, value, created_at, updated_at FROM memory ORDER BY updated_at DESC, key',
            ).fetchall()
            items = [self._item(row) for row in rows]
            matches = [item for item in items
                       if item.namespace[:len(op.namespace_prefix)] == op.namespace_prefix
                       and all(item.value.get(key) == value for key, value in (op.filter or {}).items())]
            selected = matches[op.offset:op.offset + op.limit]
            for item in selected:
                self._audit(item.namespace, item.key, 'read', reason)
            return [SearchItem(item.namespace, item.key, item.value, item.created_at, item.updated_at) for item in selected]
        raise NotImplementedError(f'Операция Store {type(op).__name__} не поддерживается.')

    def put_episode(self, client_id, key, value):
        """Эпизод, квитанция и задание индексации коммитятся вместе."""
        namespace = ('clients', client_id, 'episodes')
        with self.lock, self.connection:
            if self.connection.execute('SELECT 1 FROM memory WHERE namespace=? AND key=?',
                                       (json.dumps(namespace), key)).fetchone():
                return
            token = self.client_token(client_id)
            if self.connection.execute('SELECT 1 FROM memory_receipts WHERE client_token=? AND request_id=?',
                                       (token, key)).fetchone():
                return
            self._apply(PutOp(namespace, key, value))
            now = self.clock()
            self.connection.execute('INSERT INTO memory_receipts VALUES (?, ?, ?, ?, ?, ?)', (
                token, key, 'withdraw', 'completed', now.isoformat(),
                (now + timedelta(days=self.retention_days)).isoformat(),
            ))

    def audit(self, client_id):
        with self.lock:
            rows = self.connection.execute('''SELECT operation, role, key_token, occurred_at, retain_until, reason
                FROM memory_audit WHERE client_token=? ORDER BY id''', (self.client_token(client_id),)).fetchall()
        return [dict(zip(('operation', 'role', 'key_token', 'occurred_at', 'retain_until', 'reason'), row)) for row in rows]

    def receipts(self, client_id):
        with self.lock:
            rows = self.connection.execute('''SELECT request_id, action, status, completed_at, retain_until
                FROM memory_receipts WHERE client_token=?''', (self.client_token(client_id),)).fetchall()
        return [dict(zip(('request_id', 'action', 'status', 'completed_at', 'retain_until'), row)) for row in rows]

    def forget(self, client_id):
        """Удаляет пользовательские записи и производные данные, оставляя служебный минимум."""
        token = self.client_token(client_id)
        owned_files = [path for path in self.artifact_root.glob(f'{token}_*.json')
                       if path.resolve().parent == self.artifact_root]
        with self.lock, self.connection:
            for table in ('memory_audit', 'memory_receipts'):
                self.connection.execute(f'DELETE FROM {table} WHERE client_token=? AND retain_until<=?',
                                        (token, self.clock().isoformat()))
            rows = self.connection.execute('SELECT namespace, key FROM memory').fetchall()
            selected = [(tuple(json.loads(ns)), key) for ns, key in rows
                        if tuple(json.loads(ns))[:2] == ('clients', client_id)]
            for namespace, key in selected:
                self._apply(PutOp(namespace, key, None), reason='user_erasure')
            for table in ('memory_vectors', 'memory_jobs'):
                for namespace, key in self.connection.execute(f'SELECT namespace, key FROM {table}').fetchall():
                    if tuple(json.loads(namespace))[:2] == ('clients', client_id):
                        self.connection.execute(f'DELETE FROM {table} WHERE namespace=? AND key=?', (namespace, key))
            self.connection.execute('UPDATE memory_artifacts SET delete_pending=1 WHERE client_token=?', (token,))
            for path in owned_files:
                self.connection.execute('''INSERT OR REPLACE INTO memory_artifacts
                    (client_token, path, delete_pending) VALUES (?, ?, 1)''', (token, str(path)))
            self._audit(('clients', client_id, 'privacy'), 'erasure', 'erase', 'user_request')
            retained = self.connection.execute('SELECT COUNT(*) FROM memory_receipts WHERE client_token=?',
                                               (token,)).fetchone()[0]
        return {'deleted_records': len(selected), 'retained_receipts': retained,
                'retention_days': self.retention_days}

    def register_artifact(self, client_id, path):
        path = Path(path).resolve()
        if path.parent != self.artifact_root or path.suffix != '.json':
            raise ValueError('Трассировка находится вне каталога RAG.')
        try:
            with self.lock, self.connection:
                self.connection.execute('''INSERT OR REPLACE INTO memory_artifacts
                    (client_token, path, delete_pending, expires_at) VALUES (?, ?, 0, ?)''',
                    (self.client_token(client_id), str(path), (self.clock() + timedelta(days=30)).isoformat()))
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def cleanup_artifacts(self):
        with self.lock, self.connection:
            self.connection.execute("UPDATE memory_artifacts SET delete_pending=1 WHERE expires_at<>'' AND expires_at<=?",
                                    (self.clock().isoformat(),))
            rows = self.connection.execute('SELECT client_token, path FROM memory_artifacts WHERE delete_pending=1').fetchall()
            for token, name in rows:
                path = Path(name)
                if path.resolve().parent != self.artifact_root or path.suffix != '.json':
                    raise ValueError('Нельзя удалять файл вне каталога трассировок.')
                path.unlink(missing_ok=True)
                self.connection.execute('DELETE FROM memory_artifacts WHERE client_token=? AND path=?', (token, name))
        return len(rows)

    def purge_expired(self):
        now = self.clock()
        removed = 0
        with self.lock, self.connection:
            for namespace, key, value in self.connection.execute('SELECT namespace, key, value FROM memory').fetchall():
                data = json.loads(value)
                if data.get('expires_at') and parse_timestamp(data['expires_at']) <= now:
                    self._apply(PutOp(tuple(json.loads(namespace)), key, None), reason='expired')
                    removed += 1
            for table in ('memory_audit', 'memory_receipts'):
                self.connection.execute(f'DELETE FROM {table} WHERE retain_until<=?', (now.isoformat(),))
        return removed

    def index_status(self, client_id):
        """Возвращает число готовых, ожидающих и исчерпавших повторы записей клиента."""
        ready, pending, failed = 0, 0, 0
        model = self.embeddings.model if self.embeddings is not None else ''
        now = self.clock()
        with self.lock:
            rows = self.connection.execute('''SELECT m.namespace, m.value, v.model, j.attempts
                FROM memory m LEFT JOIN memory_vectors v
                ON m.namespace=v.namespace AND m.key=v.key AND m.revision=v.revision
                LEFT JOIN memory_jobs j ON m.namespace=j.namespace AND m.key=j.key''').fetchall()
            for namespace, value, indexed_model, attempts in rows:
                record = json.loads(value)
                if tuple(json.loads(namespace))[:2] != ('clients', client_id):
                    continue
                if not parse_timestamp(record['created_at']) <= now < parse_timestamp(record['expires_at']):
                    continue
                if indexed_model == model:
                    ready += 1
                else:
                    pending += 1
                    failed += int((attempts or 0) >= 3)
        return {'ready': ready, 'pending': pending, 'failed': failed}

    @staticmethod
    def _item(row):
        namespace, key, value, created_at, updated_at = row
        return Item(namespace=tuple(json.loads(namespace)), key=key, value=json.loads(value),
                    created_at=parse_timestamp(created_at), updated_at=parse_timestamp(updated_at))

    def _semantic(self, op):
        """Ищет действующие записи одного клиента по близости и свежести, записывая чтение в аудит."""
        if self.embeddings is None:
            raise NotImplementedError('Для поиска по смыслу подключите embeddings.')
        if len(op.namespace_prefix) < 2 or op.namespace_prefix[0] != 'clients':
            raise ValueError('Поиск по памяти должен быть ограничен одним клиентом.')
        vector = np.asarray(self.embeddings.embed_query(op.query), dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all() or not np.linalg.norm(vector):
            raise ValueError('Некорректный вектор запроса памяти.')
        now, ranked = self.clock(), []
        with self.lock, self.connection:
            rows = self.connection.execute('''SELECT m.namespace, m.key, m.value, m.created_at, m.updated_at, v.vector
                FROM memory m JOIN memory_vectors v ON m.namespace=v.namespace AND m.key=v.key AND m.revision=v.revision
                WHERE v.model=?''', (self.embeddings.model,)).fetchall()
            for *fields, blob in rows:
                item = self._item(fields)
                if item.namespace[:len(op.namespace_prefix)] != op.namespace_prefix:
                    continue
                if not all(item.value.get(key) == value for key, value in (op.filter or {}).items()):
                    continue
                created = parse_timestamp(item.value['created_at'])
                expires = parse_timestamp(item.value['expires_at'])
                if not created <= now < expires:
                    continue
                candidate = np.frombuffer(blob, dtype=np.float32)
                if candidate.shape != vector.shape or not np.isfinite(candidate).all() or not np.linalg.norm(candidate):
                    raise ValueError('Размерность памяти не совпадает с моделью запроса.')
                similarity = float(np.dot(vector, candidate) / (np.linalg.norm(vector) * np.linalg.norm(candidate)))
                similarity = max(-1.0, min(1.0, similarity))
                if similarity < self.min_similarity:
                    continue
                age_days = (now - created).total_seconds() / 86400
                freshness = math.exp(-math.log(2) * age_days / 30)
                score = 0.8 * similarity + 0.2 * freshness
                value = {**item.value, '_retrieval': {'similarity': similarity, 'freshness': freshness}}
                ranked.append(SearchItem(item.namespace, item.key, value, item.created_at, item.updated_at, score))
            ranked.sort(key=lambda item: (-item.score, item.key))
            selected = ranked[op.offset:op.offset + op.limit]
            for item in selected:
                self._audit(item.namespace, item.key, 'search', 'semantic_and_recency')
        return selected

    def close(self):
        with self.lock:
            self.connection.close()
