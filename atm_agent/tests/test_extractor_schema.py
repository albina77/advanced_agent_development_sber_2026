import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('pydantic'), 'Нужен pydantic из requirements.txt')
class ExtractorSchemaTest(unittest.TestCase):
    def test_empty_model_notes_mean_no_banknote_composition(self):
        from atm_agent.understanding.extractor import ParsedTurn

        parsed = ParsedTurn.model_validate({'relevance': 'addressed', 'intent': 'confirm',
                                           'amount': 9000, 'denomination': 1000, 'notes': []})
        self.assertIsNone(parsed.to_turn().notes)

    def test_current_turn_evidence_rejects_context_echo_but_keeps_explicit_zero(self):
        from atm_agent.understanding.extractor import ParsedTurn

        echoed = ParsedTurn(relevance='addressed', intent='resume', amount=7000, denomination=0,
                            atm_choice=True, amount_quote='сумму')
        turn = echoed.to_turn('Сумму я уже называла, купюры выберу позже.')
        self.assertIsNone(turn.amount)
        self.assertIsNone(turn.denomination)
        self.assertFalse(turn.atm_choice)
        explicit = ParsedTurn(relevance='addressed', intent='withdraw', amount=0, amount_quote='ноль рублей')
        self.assertEqual(explicit.to_turn('Сними ноль рублей').amount, 0)
        denomination = ParsedTurn(relevance='addressed', intent='details', denomination=1000,
                                  denomination_quote='Тысячными')
        self.assertEqual(denomination.to_turn('тысячными').denomination, 1000)
        whitespace = ParsedTurn(relevance='addressed', intent='details', atm_choice=True, atm_choice_quote=' ')
        self.assertFalse(whitespace.to_turn('Я ещё думаю').atm_choice)

    def test_exact_banknotes_decode(self):
        from atm_agent.understanding.extractor import ParsedTurn

        parsed = ParsedTurn.model_validate({
            'relevance': 'addressed', 'intent': 'withdraw', 'amount': 7000,
            'notes': [{'denomination': 5000, 'count': 1}, {'denomination': 1000, 'count': 2}],
        })
        self.assertEqual(parsed.to_turn().notes, ((5000, 1), (1000, 2)))

    def test_invalid_types_and_unknown_intent_are_rejected(self):
        from pydantic import ValidationError

        from atm_agent.understanding.extractor import ParsedTurn

        for extra in [{'amount': True}, {'amount': 7000.5}, {'intent': 'execute_anything'}]:
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                ParsedTurn.model_validate({'relevance': 'addressed', 'intent': 'withdraw', **extra})

    def test_preference_extraction_uses_only_current_text_and_sums_actual_usage(self):
        from langchain_core.messages import AIMessage

        from atm_agent.understanding.extractor import (
            GigaChatExtractor,
            ParsedPreference,
            ParsedTurn,
        )

        calls = []

        class Model:
            def with_structured_output(self, schema, **kwargs):
                class Structured:
                    def invoke(self, messages):
                        calls.append((schema, messages))
                        if schema is ParsedTurn:
                            parsed = ParsedTurn(relevance='addressed', intent='preference')
                        else:
                            parsed = ParsedPreference(action='set', denomination=500)
                        return {'parsed': parsed, 'raw': AIMessage(content='', usage_metadata={
                            'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12,
                        })}

                return Structured()

        extractor = GigaChatExtractor(Model())
        turn, usage = extractor.extract('Запомни: обычно по 500', {'history': [
            {'role': 'user', 'content': 'Сними тысячными'},
        ]})
        self.assertEqual(turn.denomination, 500)
        self.assertEqual(turn.intent, 'preference')
        self.assertEqual(calls[1][1][-1].content, 'Запомни: обычно по 500')
        self.assertNotIn('Сними тысячными', calls[1][1][0].content + calls[1][1][1].content)
        self.assertEqual(usage['total_tokens'], 24)

    def test_background_preference_never_invokes_memory_extractor(self):
        from langchain_core.messages import AIMessage

        from atm_agent.understanding.extractor import GigaChatExtractor, ParsedTurn

        class Model:
            def with_structured_output(self, schema, **kwargs):
                class Structured:
                    def invoke(self, messages):
                        if schema is not ParsedTurn:
                            raise AssertionError('Не извлекать память из чужой речи')
                        return {'parsed': ParsedTurn(relevance='background', intent='preference'),
                                'raw': AIMessage(content='')}

                return Structured()

        turn, _ = GigaChatExtractor(Model()).extract('Посторонняя речь', {})
        self.assertEqual(turn.relevance, 'background')
