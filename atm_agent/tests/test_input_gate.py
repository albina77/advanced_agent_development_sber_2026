import unittest

from atm_agent.atm.contracts import ATMCommand, ATMResult
from atm_agent.atm.mock import MockATMSoftware
from atm_agent.dialogue.state import Turn, handle_turn
from atm_agent.understanding.input_gate import Relevance, dispatch_turn


class InputGateTest(unittest.TestCase):
    def test_background_and_uncertainty_preserve_offer_and_do_not_confirm(self):
        software = MockATMSoftware({})
        state = handle_turn({}, Turn(Relevance.ADDRESSED, 'withdraw', amount=7000, denomination=1000))
        offer = state['pending']
        request_id = offer['request_id']
        software.responses[request_id] = ATMResult(request_id, 'completed', amount=7000)
        handled = []

        def handle(text):
            handled.append(text)
            state.update(handle_turn(state, Turn(Relevance.ADDRESSED, 'confirm')))
            return software.send(ATMCommand(**state['outgoing']))

        for relevance, text in [
            (Relevance.BACKGROUND, 'Да, бери семь тысяч, я тебе потом переведу'),
            (Relevance.BACKGROUND, 'Объявляется посадка на рейс'),
            (Relevance.UNCERTAIN, 'Да'),
        ]:
            with self.subTest(relevance=relevance, text=text):
                self.assertIsNone(dispatch_turn(text, relevance, handle))
                self.assertIs(state['pending'], offer)
                self.assertEqual(software.commands, [])
                self.assertEqual(handled, [])

        result = dispatch_turn('Да, выдайте семь тысяч тысячными', Relevance.ADDRESSED, handle)
        self.assertEqual(result.status, 'completed')
        self.assertEqual(software.commands, [ATMCommand(**offer)])
        self.assertEqual(handled, ['Да, выдайте семь тысяч тысячными'])

    def test_ignored_turn_does_not_reach_profile_or_history_writer(self):
        profile = {}
        history = []

        def handle(text):
            history.append(text)
            profile['preferred_denomination'] = 5000
            return 'Запомнила'

        self.assertIsNone(dispatch_turn('Я всегда беру пятитысячными', Relevance.BACKGROUND, handle))
        self.assertEqual(profile, {})
        self.assertEqual(history, [])
