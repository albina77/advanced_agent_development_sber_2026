import asyncio
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from atm_agent.atm.contracts import ATMCommand, ATMResult
from atm_agent.atm.mock import DemoATMSoftware
from atm_agent.dialogue.graph import build_graph
from atm_agent.dialogue.state import Turn, command_state
from atm_agent.memory.client import ClientMemory, MemoryRecord
from atm_agent.memory.store import SqliteStore
from atm_agent.understanding.input_gate import Relevance


class Extractor:
    def extract(self, text, state):
        turns = {
            'снять': Turn(Relevance.ADDRESSED, 'withdraw', amount=7000),
            'другая сумма': Turn(Relevance.ADDRESSED, 'withdraw', amount=8500),
            'без суммы': Turn(Relevance.ADDRESSED, 'withdraw'),
            'тысячными': Turn(Relevance.ADDRESSED, 'details', denomination=1000),
            'нет, тысячными': Turn(Relevance.ADDRESSED, 'reject', denomination=1000),
            'да': Turn(Relevance.ADDRESSED, 'confirm'),
            'нет': Turn(Relevance.ADDRESSED, 'reject'),
            'баланс': Turn(Relevance.ADDRESSED, 'balance'),
            'запомни': Turn(Relevance.ADDRESSED, 'preference', denomination=500),
            'забудь': Turn(Relevance.ADDRESSED, 'preference', preference_action='clear'),
            'не меняй': Turn(Relevance.ADDRESSED, 'preference', preference_action='keep'),
            'фон': Turn(Relevance.BACKGROUND, 'preference', denomination=500),
            'отмена': Turn(Relevance.ADDRESSED, 'cancel'),
        }
        return turns[text], {}


class MemoryTest(unittest.TestCase):
    def test_keep_preference_does_not_delete_or_refresh_record(self):
        self.invoke('запомни')
        before = self.store.get(('clients', 'albina', 'profile'), 'cash_preference').value
        self.now += timedelta(days=1)
        state = self.invoke('не меняй')
        after = self.store.get(('clients', 'albina', 'profile'), 'cash_preference').value
        self.assertEqual(after, before)
        self.assertEqual(self.memory.choice('albina')['denomination'], 500)
        self.assertEqual(self.software.commands, [])
        self.assertIn('без изменений', state['response'])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'memory.sqlite3'
        self.now = datetime(2026, 10, 29, tzinfo=timezone.utc)
        self.store = SqliteStore(self.path)
        self.memory = ClientMemory(self.store, clock=lambda: self.now)
        self.software = DemoATMSoftware({})
        self.app = build_graph(Extractor(), self.software, memory=self.memory)
        self.config = {'configurable': {'thread_id': 'one', 'client_id': 'albina'}}

    def tearDown(self):
        self.store.close()

    def invoke(self, text, config=None):
        return self.app.invoke({'text': text}, config or self.config)

    def seed_episode(self, denomination=1000, amount=7000):
        command = command_state(ATMCommand('request-1', 'withdraw', amount, denomination))
        result = vars(ATMResult('request-1', 'completed', amount=amount))
        self.memory.record_withdrawal('albina', command, result)

    def test_new_session_recall_requires_two_confirmations(self):
        for text in ('снять', 'тысячными', 'да'):
            self.invoke(text)
        self.assertEqual(len(self.software.commands), 1)
        config = {'configurable': {'thread_id': 'two', 'client_id': 'albina'}}
        state = self.invoke('снять', config)
        self.assertEqual(state['cash_offer']['kind'], 'withdrawal_episode')
        self.assertFalse(state['confirmation_active'])
        self.assertIn('Запомни', state['response'])
        state = self.invoke('да', config)
        self.assertTrue(state['confirmation_active'])
        self.assertIsNone(state['cash_offer'])
        self.assertEqual(len(self.software.commands), 1)
        self.invoke('да', config)
        self.assertEqual(len(self.software.commands), 2)

    def test_model_echo_on_offer_confirmation_selects_cash_without_executing(self):
        self.seed_episode()
        self.invoke('снять')

        class EchoExtractor:
            def extract(self, text, state):
                return Turn(Relevance.ADDRESSED, 'confirm', amount=7000, denomination=1000, notes=()), {}

        self.app = build_graph(EchoExtractor(), self.software, memory=self.memory)
        # Восстанавливаем текущее предложение как отдельный вход тестового графа.
        state = self.app.invoke({'text': 'да', 'slots': {'action': 'withdraw', 'amount': 7000},
                                 'cash_offer': self.memory.choice('albina'), 'cash_offer_active': True}, self.config)
        self.assertTrue(state['confirmation_active'])
        self.assertEqual(self.software.commands, [])
        self.invoke('да')
        self.assertEqual(len(self.software.commands), 1)

    def test_preference_survives_store_reopen_and_wins_over_recent_episode(self):
        self.invoke('запомни')
        self.now += timedelta(hours=1)
        self.seed_episode()
        self.store.close()
        self.store = SqliteStore(self.path)
        self.memory = ClientMemory(self.store, clock=lambda: self.now)
        choice = self.memory.choice('albina')
        self.assertEqual(choice['kind'], 'cash_preference')
        self.assertEqual(choice['denomination'], 500)
        self.assertEqual(self.software.commands, [])

    def test_one_off_choice_does_not_change_profile(self):
        self.invoke('запомни')
        self.invoke('снять')
        state = self.invoke('нет, тысячными')
        self.assertEqual(state['pending']['denomination'], 1000)
        self.invoke('да')
        self.assertEqual(self.memory.choice('albina')['denomination'], 500)
        self.assertEqual(self.software.commands[0].denomination, 1000)

    def test_new_explicit_preference_replaces_old_and_clear_preserves_episodes(self):
        self.memory.set_preference('albina', 1000)
        self.invoke('запомни')
        self.assertEqual(self.memory.choice('albina')['denomination'], 500)
        self.seed_episode()
        self.invoke('забудь')
        self.assertEqual(self.memory.choice('albina')['kind'], 'withdrawal_episode')
        self.assertEqual(self.software.commands, [])

    def test_expired_episode_and_preference_are_not_used(self):
        self.seed_episode()
        self.now += timedelta(days=30)
        self.assertIsNone(self.memory.choice('albina'))
        self.memory.set_preference('albina', 500)
        self.now += timedelta(days=180)
        self.assertIsNone(self.memory.choice('albina'))

    def test_read_does_not_extend_lifetime_and_expired_profile_can_fall_back(self):
        self.memory.set_preference('albina', 500)
        self.now += timedelta(days=179)
        self.seed_episode()
        self.assertEqual(self.memory.choice('albina')['denomination'], 500)
        self.now += timedelta(days=1)
        self.assertEqual(self.memory.choice('albina')['kind'], 'withdrawal_episode')

    def test_client_identity_is_trusted_config_not_text_state(self):
        self.seed_episode()
        config = {'configurable': {'thread_id': 'other', 'client_id': 'boris'}}
        with self.assertRaises(ValueError):
            self.app.invoke({'text': 'снять', 'client_id': 'albina'}, config)
        config['configurable']['thread_id'] = 'another'
        state = self.invoke('снять', config)
        self.assertEqual(state['client_id'], 'boris')
        self.assertIsNone(state.get('cash_offer'))
        self.assertIsNone(self.memory.choice('boris'))

    def test_identity_cannot_change_inside_checkpoint_thread(self):
        self.invoke('снять')
        config = {'configurable': {'thread_id': 'one', 'client_id': 'boris'}}
        with self.assertRaises(ValueError):
            self.invoke('да', config)
        self.assertEqual(self.software.commands, [])

    def test_preference_is_readable_in_another_python_process(self):
        self.memory.set_preference('albina', 500)
        source = '''import sys
from pathlib import Path
from atm_agent.memory.store import SqliteStore
from atm_agent.memory.client import ClientMemory
store = SqliteStore(Path(sys.argv[1]))
record = store.get(ClientMemory.namespace('albina', 'profile'), 'cash_preference')
print(record.value['denomination'])
store.close()
'''
        result = subprocess.run([sys.executable, '-c', source, str(self.path)],
                                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), '500')

    def test_offer_expiring_before_yes_requires_manual_selection(self):
        self.seed_episode()
        self.invoke('снять')
        self.now += timedelta(days=30)
        state = self.invoke('да')
        self.assertIsNone(state['cash_offer'])
        self.assertIsNone(state['pending'])
        self.assertFalse(state['confirmation_active'])
        self.assertEqual(self.software.commands, [])

    def test_background_never_writes_preference_or_history(self):
        state = self.invoke('фон')
        self.assertIsNone(state['response'])
        self.assertIsNone(self.memory.choice('albina'))
        self.assertEqual(state.get('history', []), [])

    def test_background_during_offer_does_not_consume_confirmation(self):
        self.seed_episode()
        self.invoke('снять')
        self.invoke('фон')
        self.invoke('да')
        self.assertEqual(self.software.commands, [])
        self.invoke('да')
        self.assertEqual(len(self.software.commands), 1)

    def test_balance_interrupt_requires_reasking_offer_then_operation(self):
        self.seed_episode()
        self.invoke('снять')
        self.invoke('баланс')
        state = self.invoke('да')
        self.assertTrue(state['cash_offer_active'])
        self.assertFalse(state['confirmation_active'])
        state = self.invoke('да')
        self.assertTrue(state['confirmation_active'])
        self.assertEqual([c.action for c in self.software.commands], ['get_balance'])
        self.invoke('да')
        self.assertEqual([c.action for c in self.software.commands], ['get_balance', 'withdraw'])

    def test_rejected_offer_is_not_proposed_again_and_can_be_changed(self):
        self.seed_episode()
        self.invoke('снять')
        self.invoke('нет')
        state = self.invoke('другая сумма')
        self.assertIsNone(state['cash_offer'])
        self.assertFalse(state['confirmation_active'])

    def test_memory_never_supplies_amount_or_incompatible_denomination(self):
        self.seed_episode()
        state = self.invoke('без суммы')
        self.assertNotIn('amount', state['slots'])
        self.assertIsNone(state['pending'])
        state = self.invoke('другая сумма')
        self.assertIsNone(state['cash_offer'])
        self.assertIsNone(state['pending'])

    def test_memory_failure_does_not_turn_completed_withdrawal_into_unknown_status(self):
        def fail(*args):
            raise OSError('failure')

        self.memory.record_withdrawal = fail
        for text in ('снять', 'тысячными', 'да'):
            state = self.invoke(text)
        self.assertEqual(state['result']['status'], 'completed')
        self.assertIsNone(state['device_pending'])
        self.assertIn('write_failed', state['memory_error'])
        self.assertEqual(len(self.software.commands), 1)

    def test_failed_pending_and_mismatched_results_are_not_episodes(self):
        command = command_state(ATMCommand('request', 'withdraw', 7000, 1000))
        for result in [ATMResult('request', 'failed'), ATMResult('request', 'pending'),
                       ATMResult('other', 'completed', amount=7000),
                       ATMResult('request', 'completed', amount=6000)]:
            self.memory.record_withdrawal('albina', command, vars(result))
        self.assertIsNone(self.memory.choice('albina'))

    def test_idempotent_episode_does_not_refresh_ttl(self):
        self.seed_episode()
        self.now += timedelta(days=29)
        self.seed_episode()
        self.now += timedelta(days=1)
        self.assertIsNone(self.memory.choice('albina'))

    def test_store_async_api_and_explicitly_unsupported_semantic_query(self):
        async def run():
            await self.store.aput(('test',), 'key', {'kind': 'fact'})
            return await self.store.aget(('test',), 'key')

        self.assertEqual(asyncio.run(run()).value, {'kind': 'fact'})
        with self.assertRaises(NotImplementedError):
            self.store.search(('test',), query='fact')

    def test_schema_rejects_inferred_preference_and_expiry_without_timezone(self):
        for patch in [{'source': 'atm_completed'}, {'amount': 7000}, {'denomination': -1},
                      {'created_at': self.now.replace(tzinfo=None)}]:
            data = dict(kind='cash_preference', source='user_explicit', denomination=1000,
                        created_at=self.now, expires_at=self.now + timedelta(days=180))
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                MemoryRecord(**(data | patch))
