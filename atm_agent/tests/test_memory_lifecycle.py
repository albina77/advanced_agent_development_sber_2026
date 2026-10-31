import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from atm_agent.atm.contracts import ATMCommand, ATMResult
from atm_agent.atm.mock import DemoATMSoftware
from atm_agent.dialogue.graph import build_graph
from atm_agent.dialogue.state import Turn, command_state
from atm_agent.memory.client import ClientMemory
from atm_agent.memory.store import SqliteStore
from atm_agent.memory.worker import claim_batch, process_batch, run_once
from atm_agent.understanding.input_gate import Relevance


class Embeddings:
    model = 'test-memory-embeddings'

    def __init__(self):
        self.calls = 0
        self.callback = None

    def embed_documents(self, texts):
        self.calls += 1
        if self.callback:
            self.callback()
        return [[1., 0.] for _ in texts]

    def embed_query(self, text):
        return [1., 0.]


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'store.sqlite3'
        self.artifacts = Path(self.folder.name) / 'traces'
        self.artifacts.mkdir()
        self.now = datetime(2026, 10, 29, tzinfo=timezone.utc)
        self.embeddings = Embeddings()
        self.store = SqliteStore(self.path, embeddings=self.embeddings, clock=lambda: self.now,
                                 artifact_root=self.artifacts)
        self.memory = ClientMemory(self.store, clock=lambda: self.now)
        self.addCleanup(self.store.close)

    def count(self, table):
        return self.store.connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]

    def episode(self, client='albina', request='request-1'):
        command = command_state(ATMCommand(request, 'withdraw', 7000, 1000))
        result = vars(ATMResult(request, 'completed', amount=7000))
        self.memory.record_withdrawal(client, command, result)
        return command, result

    def test_write_is_audited_and_queued_without_calling_embeddings(self):
        self.memory.set_preference('albina', 500)
        self.assertEqual(self.embeddings.calls, 0)
        self.assertEqual(self.count('memory_jobs'), 1)
        audit = self.store.audit('albina')
        self.assertEqual(audit[0]['operation'], 'create')
        text = json.dumps(audit)
        self.assertNotIn('denomination', text)
        self.assertNotIn('albina', text)
        self.assertNotIn('"500"', text)
        self.memory.set_preference('albina', 1000)
        self.assertEqual(self.store.audit('albina')[-1]['operation'], 'update')
        self.assertEqual(self.count('memory_jobs'), 1)

    def test_write_queue_and_audit_roll_back_together(self):
        with patch.object(self.store, '_audit', side_effect=OSError('failure')):
            with self.assertRaises(OSError):
                self.memory.set_preference('albina', 500)
        for table in ('memory', 'memory_jobs', 'memory_audit'):
            self.assertEqual(self.count(table), 0)

    def test_worker_indexes_only_outside_dialogue_and_records_search_scores(self):
        self.memory.set_preference('albina', 500)
        self.assertEqual(self.store.index_status('albina')['pending'], 1)
        before = self.memory.recall('albina', 'Что я обычно выбираю?')
        self.assertIn('ещё не подготовлена', before['response'])
        self.assertEqual(before['matches'], [])
        report = run_once(self.store, self.embeddings)
        self.assertEqual(report['indexed'], 1)
        self.assertEqual(self.store.index_status('albina')['ready'], 1)
        result = self.memory.recall('albina', 'Что я обычно выбираю?')
        self.assertIn('500', result['response'])
        self.assertGreater(result['matches'][0]['similarity'], 0.9)
        self.assertIn('freshness', result['matches'][0])
        self.assertEqual(self.store.audit('albina')[-1]['operation'], 'search')

    def test_freshness_changes_order_when_similarity_is_equal(self):
        self.episode(request='old')
        self.now += timedelta(days=10)
        self.episode(request='new')
        run_once(self.store, self.embeddings)
        items = self.store.search(('clients', 'albina'), query='Прошлая выдача')
        self.assertEqual([item.key for item in items], ['new', 'old'])
        self.assertGreater(items[0].score, items[1].score)

    def test_semantic_search_excludes_other_clients_expired_and_unrelated_vectors(self):
        self.episode('albina', 'old')
        self.episode('boris', 'private')
        self.memory.set_preference('albina', 500)
        run_once(self.store, self.embeddings)
        self.now += timedelta(days=30)
        ns = json.dumps(('clients', 'albina', 'profile'))
        with self.store.connection:
            self.store.connection.execute('UPDATE memory_vectors SET vector=? WHERE namespace=?',
                                          (np.asarray([0., 1.], dtype=np.float32).tobytes(), ns))
        self.assertEqual(self.store.search(('clients', 'albina'), query='Запрос'), [])
        with self.assertRaises(ValueError):
            self.store.search(('clients',), query='Другой клиент')

    def test_forget_erases_vectors_jobs_and_records_but_retains_minimal_receipts(self):
        command, result = self.episode()
        self.memory.set_preference('albina', 500)
        self.memory.set_preference('boris', 1000)
        run_once(self.store, self.embeddings)
        report = self.memory.forget('albina')
        self.assertEqual(report['deleted_records'], 2)
        self.assertEqual(report['retained_receipts'], 1)
        self.assertIsNone(self.memory.choice('albina'))
        self.assertEqual(self.store.index_status('albina'), {'ready': 0, 'pending': 0, 'failed': 0})
        self.assertEqual(self.store.search(('clients', 'albina'), query='Что помнишь?'), [])
        self.assertEqual(self.memory.choice('boris')['denomination'], 1000)
        receipt = self.store.receipts('albina')[0]
        self.assertEqual(set(receipt), {'request_id', 'action', 'status', 'completed_at', 'retain_until'})
        self.memory.record_withdrawal('albina', command, result)
        self.assertIsNone(self.memory.choice('albina'))
        self.assertTrue(any(row['operation'] == 'erase' for row in self.store.audit('albina')))

    def test_erasure_during_network_call_cannot_resurrect_record(self):
        self.memory.set_preference('albina', 500)
        self.embeddings.callback = lambda: self.memory.forget('albina')
        report = run_once(self.store, self.embeddings)
        self.assertEqual(report['discarded'], 1)
        self.assertEqual(self.count('memory_vectors'), 0)
        self.assertEqual(self.count('memory_jobs'), 0)
        self.assertIsNone(self.memory.choice('albina'))

    def test_update_during_network_call_rejects_stale_vector(self):
        self.memory.set_preference('albina', 500)
        self.embeddings.callback = lambda: self.memory.set_preference('albina', 1000)
        jobs = claim_batch(self.store, self.embeddings.model, 16)
        report = process_batch(self.store, self.embeddings, jobs)
        self.assertEqual(report['discarded'], 1)
        self.assertEqual(self.count('memory_vectors'), 0)
        self.embeddings.callback = None
        self.assertEqual(run_once(self.store, self.embeddings)['indexed'], 1)
        self.assertEqual(self.memory.choice('albina')['denomination'], 1000)

    def test_two_workers_lease_jobs_and_expired_lease_cannot_overwrite_new_worker(self):
        self.memory.set_preference('albina', 500)
        other = SqliteStore(self.path, clock=lambda: self.now)
        self.addCleanup(other.close)
        first = claim_batch(self.store, self.embeddings.model, 16)
        self.assertEqual(claim_batch(other, self.embeddings.model, 16), [])
        self.now += timedelta(minutes=6)
        second = claim_batch(other, self.embeddings.model, 16)
        self.assertEqual(len(second), 1)
        self.assertEqual(process_batch(self.store, self.embeddings, first)['discarded'], 1)
        self.assertEqual(process_batch(other, self.embeddings, second)['indexed'], 1)

    def test_retry_budget_is_bounded_and_error_body_not_saved(self):
        self.memory.set_preference('albina', 500)
        def fail():
            raise ConnectionError('sensitive-response')
        self.embeddings.callback = fail
        for _ in range(4):
            run_once(self.store, self.embeddings)
            self.now += timedelta(minutes=1)
        self.assertEqual(self.embeddings.calls, 3)
        self.assertEqual(self.store.index_status('albina')['failed'], 1)
        self.assertNotIn('sensitive-response', json.dumps(self.store.audit('albina')))
        self.memory.clear_preference('albina')
        self.assertEqual(self.count('memory_jobs'), 0)

    def test_expiry_physically_removes_derived_data_and_later_releases_receipts(self):
        self.episode()
        self.memory.set_preference('albina', 500)
        run_once(self.store, self.embeddings)
        self.now += timedelta(days=180)
        report = run_once(self.store, self.embeddings)
        self.assertEqual(report['expired'], 2)
        for table in ('memory', 'memory_jobs', 'memory_vectors'):
            self.assertEqual(self.count(table), 0)
        self.assertEqual(len(self.store.receipts('albina')), 1)
        self.now += timedelta(days=185)
        run_once(self.store, self.embeddings)
        self.assertEqual(self.store.receipts('albina'), [])

    def test_artifact_cleanup_is_owned_and_cannot_delete_outside_trace_directory(self):
        own, other = self.artifacts/'one.json', self.artifacts/'two.json'
        own.write_text('private history')
        other.write_text('other history')
        self.store.register_artifact('albina', own)
        self.store.register_artifact('boris', other)
        with self.assertRaises(ValueError):
            self.store.register_artifact('albina', Path(self.folder.name)/'outside.json')
        self.memory.forget('albina')
        self.assertEqual(self.store.cleanup_artifacts(), 1)
        self.assertFalse(own.exists())
        self.assertTrue(other.exists())

    def test_trace_expiry_and_unregistered_owned_trace_are_cleaned(self):
        trace = self.artifacts/'registered.json'
        trace.write_text('private history')
        self.store.register_artifact('albina', trace)
        self.now += timedelta(days=30)
        run_once(self.store, self.embeddings)
        self.assertFalse(trace.exists())
        orphan = self.artifacts / (self.store.client_token('albina') + '_lost.json')
        orphan.write_text('private history')
        self.memory.forget('albina')
        self.store.cleanup_artifacts()
        self.assertFalse(orphan.exists())

    def test_model_change_requires_reindex_and_does_not_mix_vector_spaces(self):
        self.memory.set_preference('albina', 500)
        run_once(self.store, self.embeddings)
        self.embeddings.model = 'changed-model'
        self.assertEqual(self.store.index_status('albina')['pending'], 1)
        self.assertEqual(self.store.search(('clients', 'albina'), query='Предпочтение'), [])
        self.assertEqual(run_once(self.store, self.embeddings)['indexed'], 1)
        self.assertEqual(self.store.index_status('albina')['ready'], 1)

    def test_graph_erasure_clears_all_client_checkpoints_and_cancels_proposals(self):
        class Extractor:
            def extract(self, text, state):
                return {
                    'снять': Turn(Relevance.ADDRESSED, 'withdraw', amount=7000, denomination=1000),
                    'удалить': Turn(Relevance.ADDRESSED, 'forget_memory'),
                    'да': Turn(Relevance.ADDRESSED, 'confirm'),
                    'фон': Turn(Relevance.BACKGROUND, 'forget_memory'),
                }[text], {}
        software = DemoATMSoftware({})
        app = build_graph(Extractor(), software, memory=self.memory)
        configs = [{'configurable': {'thread_id': thread, 'client_id': client}}
                   for thread, client in [('one','albina'), ('two','albina'), ('three','boris')]]
        self.memory.set_preference('albina', 500)
        for config in configs:
            app.invoke({'text': 'снять'}, config)
        app.invoke({'text': 'фон'}, configs[0])
        self.assertIsNotNone(self.memory.choice('albina'))
        state = app.invoke({'text': 'удалить'}, configs[0])
        self.assertIn('удалены', state['response'])
        self.assertEqual(state['deletion_report']['deleted_records'], 1)
        for config in configs[:2]:
            self.assertEqual(app.get_state(config).values, {})
        self.assertTrue(app.get_state(configs[2]).values['pending'])
        app.invoke({'text': 'да'}, configs[0])
        self.assertEqual(software.commands, [])

    def test_rag_trace_containing_client_question_is_owned_and_deleted_with_memory(self):
        from langchain_core.messages import AIMessage

        from atm_agent.retrieval.consultant import RAGConsultant

        class Model:
            model = 'fake-model'

            def bind_tools(self, *args, **kwargs):
                return self

            def invoke(self, messages):
                return AIMessage('')

        class Index:
            documents = []

        class Extractor:
            def extract(self, text, state):
                intent = 'product' if text.startswith('вопрос') else 'forget_memory'
                return Turn(Relevance.ADDRESSED, intent), {}

        app = build_graph(Extractor(), DemoATMSoftware({}), RAGConsultant(Model(), Index()), self.memory)
        config = {'configurable': {'thread_id': 'trace', 'client_id': 'albina'}}
        self.store.artifact_root = (self.artifacts / 'runs').resolve()
        with patch('atm_agent.retrieval.consultant.INDEX', self.artifacts):
            result = app.invoke({'text': 'вопрос: личное_слово', 'history': [
                {'role': 'user', 'content': 'ранняя реплика'},
            ]}, config)
        self.assertIsNone(result.get('error'))
        path = Path(result['rag_trace_file'])
        self.assertTrue(path.name.startswith(self.store.client_token('albina') + '_'))
        self.assertIn('личное_слово', path.read_text())
        app.invoke({'text': 'удалить'}, config)
        self.assertFalse(path.exists())
        self.assertNotIn('личное_слово', json.dumps(self.store.audit('albina')))

    def test_erasure_waits_for_pending_device_result(self):
        class Extractor:
            def extract(self, text, state):
                return Turn(Relevance.ADDRESSED, 'deposit' if text == 'внести' else 'forget_memory'), {}

        software = DemoATMSoftware({})
        app = build_graph(Extractor(), software, memory=self.memory)
        config = {'configurable': {'thread_id': 'device', 'client_id': 'albina'}}
        self.memory.set_preference('albina', 500)
        before = app.invoke({'text': 'внести'}, config)
        state = app.invoke({'text': 'удалить'}, config)
        self.assertEqual(state['device_pending'], before['device_pending'])
        self.assertFalse(state['erase_memory_requested'])
        self.assertEqual(self.memory.choice('albina')['denomination'], 500)
        self.assertEqual(len(software.commands), 1)

    def test_development_similarity_boundary_keeps_relevant_and_excludes_unrelated(self):
        self.memory.set_preference('albina', 500)
        run_once(self.store, self.embeddings)
        for similarity, expected in [(0.135, 1), (0.055, 0)]:
            vector = np.asarray([similarity, (1 - similarity ** 2) ** 0.5], dtype=np.float32)
            with self.store.connection:
                self.store.connection.execute('UPDATE memory_vectors SET vector=?', (vector.tobytes(),))
            self.assertEqual(len(self.store.search(('clients', 'albina'), query='Вопрос')), expected)

    def test_old_store_is_migrated_without_losing_profile_and_episode_receipt(self):
        path = Path(self.folder.name)/'legacy.sqlite3'
        self.episode()
        value = self.store.get(('clients','albina','episodes'), 'request-1').value
        with sqlite3.connect(path) as connection:
            connection.execute('''CREATE TABLE memory (namespace TEXT, key TEXT, value TEXT,
                                created_at TEXT, updated_at TEXT, PRIMARY KEY(namespace,key))''')
            connection.execute('INSERT INTO memory VALUES (?, ?, ?, ?, ?)',
                (json.dumps(('clients','albina','episodes')), 'request-1', json.dumps(value),
                 self.now.isoformat(), self.now.isoformat()))
        migrated = SqliteStore(path, clock=lambda: self.now)
        self.addCleanup(migrated.close)
        self.assertEqual(len(migrated.receipts('albina')), 1)
        self.assertIsNotNone(migrated.get(('clients','albina','episodes'), 'request-1'))
        self.assertEqual(migrated.connection.execute('SELECT COUNT(*) FROM memory_jobs').fetchone()[0], 1)
