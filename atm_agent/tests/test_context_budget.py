import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from atm_agent.atm.contracts import ATMCommand
from atm_agent.atm.mock import DemoATMSoftware
from atm_agent.context.budget import (
    ContextBudget,
    ContextBudgetError,
    history_turns,
    total_usage,
)
from atm_agent.dialogue.graph import build_graph
from atm_agent.dialogue.state import Turn, command_state, handle_turn
from atm_agent.understanding.extractor import (
    GigaChatExtractor,
    ParsedPreference,
    ParsedTurn,
)
from atm_agent.understanding.input_gate import Relevance


class Schema(BaseModel):
    action: str


class SummaryModel:
    def __init__(self, content='Клиент не закончил снятие денег на дорогу.', actual_input=20, fail=False):
        self.content, self.actual_input, self.fail = content, actual_input, fail
        self.calls = []

    def bind(self, **kwargs):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise ConnectionError('sensitive-error')
        return AIMessage(self.content, usage_metadata={
            'input_tokens': self.actual_input, 'output_tokens': 10, 'total_tokens': self.actual_input + 10,
        })


def messages(text, state):
    return [SystemMessage('Rules'), HumanMessage(json.dumps({
        'utterance': text, 'history': state.get('history', []), 'summary': state.get('summary', ''),
        'slots': state.get('slots', {}),
    }, ensure_ascii=False))]


def history(turns=5, content='Речь клиента про дорогу'):
    return [message for number in range(turns) for message in (
        {'role': 'user', 'content': content + str(number)},
        {'role': 'assistant', 'content': 'Операция ещё не выполнена.'},
    )]


class BudgetTest(unittest.TestCase):
    def test_pause_preserves_operation_but_requires_new_confirmation(self):
        pending = command_state(ATMCommand('pause', 'withdraw', 7000, 1000))
        state = {'slots': {'action': 'withdraw', 'amount': 7000, 'denomination': 1000, 'cash_choice': True},
                 'pending': pending, 'confirmation_active': True}
        updates = handle_turn(state, Turn(Relevance.ADDRESSED, 'pause'))
        paused = state | updates
        self.assertEqual(paused['pending'], pending)
        self.assertEqual(paused['slots']['amount'], 7000)
        self.assertFalse(paused['confirmation_active'])
        confirmed = handle_turn(paused, Turn(Relevance.ADDRESSED, 'confirm'))
        self.assertIsNone(confirmed['outgoing'])
        self.assertTrue(confirmed['confirmation_active'])

    def test_summary_covers_only_dropped_complete_turns_and_preserves_previous_summary(self):
        model = SummaryModel()
        controller = ContextBudget(model, input_tokens=400, summary_tokens=30, recent_turns=1,
                                   overhead=0, counter=lambda text: len(text)//10)
        state = {'text': 'Продолжим', 'history': history(), 'summary': 'Старая тема', 'slots': {'amount': 7000}}
        result = controller.prepare(state, messages, Schema)
        self.assertEqual(result['history'], state['history'][-2:])
        self.assertEqual(result['folded_messages'], 8)
        self.assertEqual(state['history'], history())
        payload = json.loads(model.calls[0][-1].content)
        self.assertEqual(payload['previous_summary'], 'Старая тема')
        self.assertEqual(payload['history_to_fold'], state['history'][:-2])
        self.assertNotIn('slots', result)
        self.assertEqual(result['summary_calls'][0]['usage']['total_tokens'], 30)

    def test_device_event_response_is_kept_with_its_user_turn(self):
        items = history(1) + [{'role': 'assistant', 'content': 'Устройство завершило внесение.'}]
        self.assertEqual(len(history_turns(items)), 1)
        self.assertEqual(len(history_turns(items)[0]), 3)

    def test_oversized_current_utterance_is_not_silently_truncated_or_sent(self):
        model = SummaryModel()
        controller = ContextBudget(model, input_tokens=100, overhead=0, counter=lambda text: len(text))
        state = {'text': 'x'*1000, 'history': history(3)}
        with self.assertRaises(ContextBudgetError):
            controller.prepare(state, messages, Schema)
        self.assertEqual(model.calls, [])
        self.assertEqual(len(state['history']), 6)

    def test_failed_summary_preserves_history_and_does_not_invent_usage(self):
        model = SummaryModel(fail=True)
        controller = ContextBudget(model, input_tokens=400, summary_tokens=30, recent_turns=1,
                                   overhead=0, counter=lambda text: len(text)//10)
        state = {'text': 'Продолжим', 'history': history()}
        with self.assertRaises(ContextBudgetError) as caught:
            controller.prepare(state, messages, Schema)
        self.assertEqual(state['history'], history())
        self.assertEqual(caught.exception.calls, [])
        self.assertNotIn('sensitive-error', str(caught.exception))

    def test_actual_summary_overrun_is_logged_and_compaction_is_not_committed(self):
        model = SummaryModel(actual_input=401)
        controller = ContextBudget(model, input_tokens=400, summary_tokens=30, recent_turns=1,
                                   overhead=0, counter=lambda text: len(text)//10)
        state = {'text': 'Продолжим', 'history': history()}
        with self.assertRaises(ContextBudgetError) as caught:
            controller.prepare(state, messages, Schema)
        self.assertFalse(caught.exception.calls[0]['within_budget'])
        self.assertEqual(total_usage(caught.exception.calls)['input_tokens'], 401)
        self.assertEqual(state['history'], history())

    def test_tokenizer_failure_does_not_fall_back_to_character_estimate(self):
        def fail(text):
            raise TimeoutError()
        model = SummaryModel()
        controller = ContextBudget(model, counter=fail)
        with self.assertRaises(ContextBudgetError):
            controller.prepare({'text': 'Да'}, messages, Schema)
        self.assertEqual(model.calls, [])

    def test_tool_schema_and_overhead_are_counted_and_cache_has_no_plaintext(self):
        inputs = []
        def counter(text):
            inputs.append(text)
            return 50
        controller = ContextBudget(SummaryModel(), input_tokens=100, overhead=25, counter=counter)
        prompt = [HumanMessage('private-message')]
        controller.check('test', prompt, [Schema])
        controller.check('test', prompt, [Schema])
        self.assertEqual(len(inputs), 1)
        self.assertIn('Schema', inputs[0])
        self.assertNotIn('private-message', repr(controller.cache))
        with self.assertRaises(ContextBudgetError):
            controller.check('test', prompt, [Schema], limit=60)

    def test_citations_come_from_folded_history_not_generated_summary(self):
        model = SummaryModel(content='Беседовали о продукте [invented@2026-01-01, с. 99] и [source_id].')
        controller = ContextBudget(model, input_tokens=500, summary_tokens=30, recent_turns=1,
                                   overhead=0, counter=lambda text: len(text)//10)
        items = history()
        items[1]['content'] = 'Ответ [sberprime-offer@2026-03-01, с. 1]'
        result = controller.prepare({'text': 'Продолжим', 'history': items,
                                    'summary_citations': ['[sberspasibo-rules@2024-10-01, с. 2]']}, messages, Schema)
        self.assertEqual(result['summary_citations'], ['[sberspasibo-rules@2024-10-01, с. 2]',
                                                      '[sberprime-offer@2026-03-01, с. 1]'])
        self.assertNotIn('invented', result['summary'])
        self.assertNotIn('[source_id]', result['summary'])


class ExtractionModel(SummaryModel):
    def __init__(self, actual_input=100):
        super().__init__(actual_input=actual_input)
        self.extract_inputs = []

    def with_structured_output(self, schema, **kwargs):
        model = self
        class Structured:
            def invoke(self, messages):
                model.extract_inputs.append(messages)
                text = messages[-1].content
                if schema is ParsedPreference:
                    parsed = ParsedPreference(action='set', denomination=500)
                else:
                    payload = json.loads(text)
                    utterance = payload['utterance']
                    if utterance == 'фон':
                        parsed = ParsedTurn(relevance='background', intent='other')
                    elif utterance == 'да':
                        parsed = ParsedTurn(relevance='addressed', intent='confirm')
                    elif utterance == 'предпочтение':
                        parsed = ParsedTurn(relevance='addressed', intent='preference')
                    elif utterance == 'удалить':
                        parsed = ParsedTurn(relevance='addressed', intent='forget_memory')
                    else:
                        parsed = ParsedTurn(relevance='addressed', intent='other')
                return {'parsed': parsed, 'raw': AIMessage('', usage_metadata={
                    'input_tokens': model.actual_input, 'output_tokens': 5, 'total_tokens': model.actual_input + 5,
                })}
        return Structured()


class ContextGraphTest(unittest.TestCase):
    def setup_graph(self, actual_input=100):
        model = ExtractionModel(actual_input)
        controller = ContextBudget(model, input_tokens=4096, summary_tokens=100, recent_turns=1,
                                   counter=lambda text: len(text)//10)
        extractor = GigaChatExtractor(model, context=controller)
        software = DemoATMSoftware({})
        return model, extractor, software, build_graph(extractor, software)

    def test_compaction_keeps_pending_command_and_cannot_grant_confirmation(self):
        model, _, software, app = self.setup_graph()
        model.content = 'Клиент якобы согласился на выдачу 9000 рублей.'
        pending = command_state(ATMCommand('pending', 'withdraw', 7000, 1000))
        config = {'configurable': {'thread_id': 'compact'}}
        state = app.invoke({'text': 'Продолжим', 'history': history(), 'pending': pending,
                            'slots': {'action': 'withdraw', 'amount': 7000, 'denomination': 1000, 'cash_choice': True},
                            'confirmation_active': False}, config)
        self.assertEqual(state['pending'], pending)
        self.assertEqual(state['slots']['amount'], 7000)
        self.assertFalse(state['confirmation_active'])
        self.assertEqual(software.commands, [])
        payload = json.loads(model.extract_inputs[-1][-1].content)
        self.assertEqual(payload['context']['summary'], model.content)
        state = app.invoke({'text': 'да'}, config)
        self.assertEqual(software.commands, [])
        self.assertTrue(state['confirmation_active'])
        app.invoke({'text': 'да'}, config)
        self.assertEqual(software.commands[0].amount, 7000)

    def test_actual_extraction_overrun_blocks_command_but_records_spent_tokens(self):
        _, _, software, app = self.setup_graph(actual_input=5000)
        pending = command_state(ATMCommand('pending', 'withdraw', 7000, 1000))
        state = app.invoke({'text': 'да', 'pending': pending, 'confirmation_active': True},
                           {'configurable': {'thread_id': 'overrun'}})
        self.assertEqual(software.commands, [])
        self.assertEqual(state['error'], 'actual_context_budget_exceeded')
        self.assertEqual(state['turn_usage']['input_tokens'], 5000)
        self.assertFalse(state['budget_log'][0]['calls'][0]['within_budget'])

    def test_usage_includes_summary_preference_and_background_without_double_counting(self):
        _, _, _, app = self.setup_graph()
        config = {'configurable': {'thread_id': 'ledger'}}
        first = app.invoke({'text': 'предпочтение', 'history': history()}, config)
        self.assertEqual(first['turn_usage']['input_tokens'], 300)
        self.assertEqual([record['stage'] for record in first['budget_log'][0]['calls']],
                         ['summarize', 'extract', 'preference'])
        state = app.invoke({'text': 'фон'}, config)
        self.assertEqual(state['session_usage']['input_tokens'], 400)
        self.assertFalse(any(item['content'] == 'фон' for item in state['history']))
        self.assertEqual(len(state['budget_log']), 2)

    def test_summary_is_scoped_to_thread(self):
        _, _, _, app = self.setup_graph()
        first = app.invoke({'text': 'Продолжим', 'history': history()}, {'configurable': {'thread_id': 'one'}})
        second = app.invoke({'text': 'Здравствуйте'}, {'configurable': {'thread_id': 'two'}})
        self.assertTrue(first['summary'])
        self.assertEqual(second['summary'], '')
        self.assertEqual(second['session_usage']['input_tokens'], 100)

    def test_prepare_failure_preserves_operation_and_does_not_invoke_extractor(self):
        model, extractor, software, app = self.setup_graph()
        extractor.context.counter = lambda text: (_ for _ in ()).throw(TimeoutError())
        pending = command_state(ATMCommand('pending', 'withdraw', 7000, 1000))
        state = app.invoke({'text': 'да', 'history': history(), 'pending': pending, 'confirmation_active': True},
                           {'configurable': {'thread_id': 'failed-prepare'}})
        self.assertEqual(state['pending'], pending)
        self.assertEqual(state['history'], history())
        self.assertEqual(model.extract_inputs, [])
        self.assertEqual(software.commands, [])
        self.assertEqual(state['turn_usage'], {})

    def test_forgetting_erases_summary_citations_and_token_counter_cache(self):
        from atm_agent.memory.client import ClientMemory
        from atm_agent.memory.store import SqliteStore

        _, extractor, software, _ = self.setup_graph()
        with tempfile.TemporaryDirectory() as folder:
            store = SqliteStore(Path(folder)/'memory.sqlite3')
            try:
                app = build_graph(extractor, software, memory=ClientMemory(store))
                config = {'configurable': {'thread_id': 'private-summary', 'client_id': 'client'}}
                state = app.invoke({'text': 'Продолжим', 'history': history(),
                                    'summary_citations': ['[sberprime-offer@2026-03-01, с. 1]']}, config)
                self.assertTrue(state['summary'])
                self.assertTrue(extractor.context.cache)
                app.invoke({'text': 'удалить'}, config)
                self.assertEqual(app.get_state(config).values, {})
                self.assertEqual(extractor.context.cache, {})
            finally:
                store.close()


class RAGBudgetTest(unittest.TestCase):
    def test_oversized_evidence_is_rejected_without_losing_cost_of_tool_selection(self):
        from atm_agent.retrieval.consultant import RAGConsultant

        class Model:
            model = 'fake-model'

            def bind_tools(self, *args, **kwargs):
                return self

            def invoke(self, messages):
                return AIMessage('', tool_calls=[{'name': 'search_knowledge', 'args': {
                    'query': 'Продукт', 'as_of': '2026-10-29',
                }, 'id': 'call', 'type': 'tool_call'}], usage_metadata={
                    'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110,
                })

            def with_structured_output(self, *args, **kwargs):
                class Generator:
                    def invoke(self, messages):
                        raise AssertionError('Нельзя отправлять слишком большой контекст доказательств')
                return Generator()

        class Index:
            documents = []

            def search(self, *args):
                return {'chunks': [{'chunk_id': 'large', 'text': 'x' * 50000}], 'trace': {}}

        model = Model()
        consultant = RAGConsultant(model, Index())
        consultant.context = ContextBudget(model, rag_tokens=1000, overhead=0, counter=lambda text: len(text)//20)
        with self.assertRaises(ContextBudgetError) as caught:
            consultant.consult('Продукт')
        self.assertEqual(total_usage(caught.exception.calls)['input_tokens'], 100)
        self.assertEqual(len(caught.exception.calls), 1)
