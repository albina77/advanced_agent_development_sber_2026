import importlib.util
import unittest

from atm_agent.atm.mock import DemoATMSoftware
from atm_agent.dialogue.state import Turn, receive_result
from atm_agent.understanding.input_gate import Relevance

HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ('langgraph', 'pydantic'))


class ScriptedExtractor:
    def __init__(self):
        self.contexts = []
        self.turns = {
            'Хочу снять': Turn(Relevance.ADDRESSED, 'withdraw'),
            'Семь тысяч': Turn(Relevance.ADDRESSED, 'details', amount=7000),
            'Объявляется посадка на рейс': Turn(Relevance.BACKGROUND),
            'Тысячными': Turn(Relevance.ADDRESSED, 'details', denomination=1000),
            'Да': Turn(Relevance.ADDRESSED, 'confirm'),
            'Баланс': Turn(Relevance.ADDRESSED, 'balance'),
            'Хочу внести десять тысяч': Turn(Relevance.ADDRESSED, 'deposit'),
        }

    def extract(self, text, state):
        self.contexts.append(dict(state))
        return self.turns[text], {}


@unittest.skipUnless(HAS_DEPS, 'Для интеграционного теста установите requirements.txt')
class GraphTest(unittest.TestCase):
    def setUp(self):
        from atm_agent.dialogue.graph import build_graph

        self.software = DemoATMSoftware({})
        self.extractor = ScriptedExtractor()
        self.app = build_graph(self.extractor, self.software)
        self.config = {'configurable': {'thread_id': 'client-1-session-1'}}

    def test_checkpoint_survives_turns_and_background_is_not_remembered(self):
        for text in ['Хочу снять', 'Семь тысяч', 'Объявляется посадка на рейс', 'Тысячными']:
            result = self.app.invoke({'text': text}, self.config)
        self.assertEqual(self.software.commands, [])
        self.assertTrue(result['confirmation_active'])
        self.assertFalse(any('посадка' in item['content'] for item in result['history']))
        result = self.app.invoke({'text': 'Да'}, self.config)
        self.assertEqual(len(self.software.commands), 1)
        self.assertEqual(self.software.commands[0].amount, 7000)
        self.assertEqual(self.software.commands[0].denomination, 1000)
        self.assertEqual(result['result']['status'], 'completed')
        self.app.invoke({'text': 'Да'}, self.config)
        self.assertEqual(len(self.software.commands), 1)

    def test_sessions_are_isolated(self):
        self.app.invoke({'text': 'Хочу снять'}, self.config)
        self.app.invoke({'text': 'Семь тысяч'}, self.config)
        another = {'configurable': {'thread_id': 'client-2-session-1'}}
        result = self.app.invoke({'text': 'Да'}, another)
        self.assertEqual(result.get('slots', {}), {})
        self.assertEqual(self.software.commands, [])

    def test_balance_reflects_withdrawal_in_current_and_new_session(self):
        for text in ['Хочу снять', 'Семь тысяч', 'Тысячными']:
            self.app.invoke({'text': text}, self.config)
        self.assertEqual(self.software.balance, 50000)
        self.app.invoke({'text': 'Да'}, self.config)
        result = self.app.invoke({'text': 'Баланс'}, self.config)
        self.assertEqual(result['result']['balance'], 43000)
        next_session = {'configurable': {'thread_id': 'client-1-session-2'}}
        result = self.app.invoke({'text': 'Баланс'}, next_session)
        self.assertEqual(result['result']['balance'], 43000)

    def test_confirmation_with_context_copied_by_model_sends_once(self):
        self.extractor.turns['Да'] = Turn(Relevance.ADDRESSED, 'confirm', amount=7000, denomination=1000)
        for text in ['Хочу снять', 'Семь тысяч', 'Объявляется посадка на рейс', 'Тысячными', 'Да']:
            result = self.app.invoke({'text': text}, self.config)
        self.assertEqual(len(self.software.commands), 1)
        self.assertEqual(self.software.commands[0].amount, 7000)
        self.assertEqual(self.software.commands[0].denomination, 1000)
        self.assertEqual(result['result']['status'], 'completed')
        self.app.invoke({'text': 'Да'}, self.config)
        self.assertEqual(len(self.software.commands), 1)

    def test_device_event_finishes_deposit(self):
        result = self.app.invoke({'text': 'Хочу внести десять тысяч'}, self.config)
        command = result['device_pending']
        self.assertIsNone(command['amount'])
        event = self.software.complete_deposit(command['request_id'], 12500)
        updates = receive_result(result, event)
        self.app.update_state(self.config, updates, as_node='remember')
        self.assertIsNone(self.app.get_state(self.config).values['device_pending'])
        result = self.app.invoke({'text': 'Баланс'}, self.config)
        self.assertEqual(result['result']['balance'], 62500)

    def test_consultation_preserves_withdrawal_and_requires_new_confirmation(self):
        from atm_agent.dialogue.graph import build_graph

        class Consultant:
            calls = 0

            def consult(self, question, history):
                self.calls += 1
                return {'response': 'Ответ по документу [source, с. 1]', 'citations': [{'source_id': 'source'}],
                        'retrieval_trace': [], 'rag_tool_calls': [], 'rag_calls': [], 'supported': True}

        consultant = Consultant()
        self.extractor.turns['Вопрос о СберПрайм'] = Turn(Relevance.ADDRESSED, 'product')
        app = build_graph(self.extractor, self.software, consultant)
        for text in ['Хочу снять', 'Семь тысяч', 'Тысячными']:
            before = app.invoke({'text': text}, self.config)
        result = app.invoke({'text': 'Вопрос о СберПрайм'}, self.config)
        self.assertEqual(result['pending'], before['pending'])
        self.assertFalse(result['confirmation_active'])
        self.assertTrue(result['citations'])
        self.assertEqual(consultant.calls, 1)
        app.invoke({'text': 'Да'}, self.config)
        self.assertEqual(self.software.commands, [])
        result = app.invoke({'text': 'Да'}, self.config)
        self.assertEqual(len(self.software.commands), 1)
        self.assertEqual(result['citations'], [])

    def test_background_never_reaches_consultation(self):
        from atm_agent.dialogue.graph import build_graph

        class Consultant:
            def consult(self, *args):
                raise AssertionError('Чужая речь не должна вызывать RAG')

        app = build_graph(self.extractor, self.software, Consultant())
        result = app.invoke({'text': 'Объявляется посадка на рейс'}, self.config)
        self.assertIsNone(result['response'])
        self.assertEqual(result.get('history', []), [])

    def test_failed_rag_does_not_send_withdrawal(self):
        from atm_agent.dialogue.graph import build_graph

        class Consultant:
            def consult(self, *args):
                raise ConnectionError('Тестовый отказ')

        self.extractor.turns['Вопрос о СберПрайм'] = Turn(Relevance.ADDRESSED, 'product')
        app = build_graph(self.extractor, self.software, Consultant())
        for text in ['Хочу снять', 'Семь тысяч', 'Тысячными']:
            before = app.invoke({'text': text}, self.config)
        result = app.invoke({'text': 'Вопрос о СберПрайм'}, self.config)
        self.assertEqual(result['pending'], before['pending'])
        self.assertEqual(result['error'], 'rag_failed:ConnectionError')
        self.assertEqual(self.software.commands, [])
