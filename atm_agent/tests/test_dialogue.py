import unittest

from atm_agent.atm.contracts import ATMResult, OperationError
from atm_agent.atm.mock import DemoATMSoftware
from atm_agent.dialogue.state import Turn, handle_turn, receive_result
from atm_agent.understanding.input_gate import Relevance


class DialogueTest(unittest.TestCase):
    def test_capability_question_does_not_start_deposit_or_clear_withdrawal(self):
        state = {'slots': {'action': 'withdraw', 'amount': 12000}}
        result = handle_turn(state, Turn(Relevance.ADDRESSED, 'capabilities'))
        self.assertIsNone(result['outgoing'])
        self.assertEqual({**state, **result}['slots']['amount'], 12000)
        self.assertNotIn('device_pending', result)

    def test_resume_uses_saved_amount_after_pause_without_new_amount(self):
        self.say('withdraw', amount=12000, denomination=500)
        self.say('pause')
        result = self.say('resume')
        self.assertIn('12000 ₽', result['response'])
        self.assertEqual(result['pending']['amount'], 12000)
        self.assertEqual(result['pending']['denomination'], 500)
        self.assertIsNone(result['outgoing'])
        self.say('confirm')
        self.assertEqual(self.state['outgoing']['amount'], 12000)

    def test_help_and_greeting_offer_product_consultation(self):
        for intent in ('greeting', 'other'):
            response = handle_turn({}, Turn(Relevance.ADDRESSED, intent))['response']
            self.assertIn('проконсультировать', response)
            self.assertIn('СберПрайм', response)

    def setUp(self):
        self.state = {}
        self.software = DemoATMSoftware({})

    def say(self, intent='other', relevance=Relevance.ADDRESSED, **kwargs):
        self.state.update(handle_turn(self.state, Turn(relevance, intent, **kwargs)))
        return self.state

    def offer(self):
        self.say('withdraw')
        self.say('details', amount=7000)
        self.say('details', denomination=1000)

    def test_incremental_slots_and_explicit_confirmation(self):
        self.assertIn('сумму', self.say('withdraw')['response'])
        self.assertIn('купюрами', self.say('details', amount=7000)['response'])
        self.assertIsNone(self.state['outgoing'])
        self.say('details', denomination=1000)
        pending = self.state['pending']
        self.assertEqual(pending['amount'], 7000)
        self.assertEqual(pending['denomination'], 1000)
        self.assertTrue(self.state['confirmation_active'])
        self.assertIsNone(self.state['outgoing'])
        self.say('confirm')
        self.assertEqual(self.state['outgoing'], pending)
        self.assertIsNone(self.state['pending'])
        self.say('confirm')
        self.assertIsNone(self.state['outgoing'])

    def test_background_does_not_change_offer_even_with_amount_and_confirmation(self):
        self.offer()
        pending = self.state['pending']
        for relevance in [Relevance.BACKGROUND, Relevance.UNCERTAIN]:
            self.say('confirm', relevance, amount=20000, denomination=5000)
            self.assertEqual(self.state['pending'], pending)
            self.assertEqual(self.state['slots']['amount'], 7000)
            self.assertTrue(self.state['confirmation_active'])
            self.assertIsNone(self.state['response'])
            self.assertIsNone(self.state['outgoing'])

    def test_confirmation_with_correction_reasks_instead_of_sending(self):
        self.offer()
        old_id = self.state['pending']['request_id']
        self.say('confirm', amount=5000)
        self.assertIn('5000', self.state['response'])
        self.assertNotEqual(self.state['pending']['request_id'], old_id)
        self.assertIsNone(self.state['outgoing'])

    def test_confirmation_with_copied_amount_and_denomination_sends_existing_offer(self):
        self.offer()
        pending = self.state['pending']
        self.say('confirm', amount=7000, denomination=1000)
        self.assertEqual(self.state['outgoing'], pending)
        self.assertIsNone(self.state['pending'])
        self.assertFalse(self.state['confirmation_active'])

    def test_copied_parameters_after_interruption_still_require_fresh_question(self):
        self.offer()
        pending = self.state['pending']
        self.say('product')
        self.say('confirm', amount=7000, denomination=1000)
        self.assertIsNone(self.state['outgoing'])
        self.assertEqual(self.state['pending'], pending)
        self.assertTrue(self.state['confirmation_active'])

    def test_confirmation_with_changed_denomination_reasks(self):
        self.offer()
        self.say('confirm', amount=7000, denomination=500)
        self.assertIsNone(self.state['outgoing'])
        self.assertIn('500 ₽', self.state['response'])

    def test_confirmation_accepts_same_mix_in_different_order(self):
        self.say('withdraw', amount=7000, notes=((5000, 1), (1000, 2)))
        pending = self.state['pending']
        self.say('confirm', amount=7000, notes=((1000, 2), (5000, 1)))
        self.assertEqual(self.state['outgoing'], pending)

    def test_confirmation_with_changed_mix_reasks(self):
        self.say('withdraw', amount=7000, notes=((5000, 1), (1000, 2)))
        self.say('confirm', amount=7000, notes=((1000, 7),))
        self.assertIsNone(self.state['outgoing'])
        self.assertTrue(self.state['confirmation_active'])

    def test_invalid_correction_cannot_confirm_previous_offer(self):
        self.offer()
        self.say('details', amount=-1)
        self.assertIsNone(self.state['pending'])
        self.say('confirm')
        self.assertIsNone(self.state['outgoing'])

    def test_product_question_preserves_slots_but_requires_fresh_confirmation(self):
        self.offer()
        pending = self.state['pending']
        self.say('product', amount=1000000)
        self.assertEqual(self.state['pending'], pending)
        self.assertEqual(self.state['slots']['amount'], 7000)
        self.assertFalse(self.state['confirmation_active'])
        self.say('confirm')
        self.assertIsNone(self.state['outgoing'])
        self.assertTrue(self.state['confirmation_active'])
        self.say('confirm')
        self.assertEqual(self.state['outgoing'], pending)

    def test_rejection_requires_new_offer(self):
        self.offer()
        self.say('reject')
        self.say('confirm')
        self.assertIsNone(self.state['outgoing'])
        self.say('details', denomination=500)
        self.assertIn('500 ₽', self.state['response'])
        self.assertIsNone(self.state['outgoing'])

    def test_cancel_clears_slots(self):
        self.offer()
        self.say('cancel')
        self.assertEqual(self.state['slots'], {})
        self.say('confirm')
        self.assertIsNone(self.state['outgoing'])

    def test_atm_choice_replaces_exact_mix(self):
        self.say('withdraw', amount=7000, notes=((5000, 1), (1000, 2)))
        self.say('details', atm_choice=True)
        self.assertIsNone(self.state['pending']['denomination'])
        self.assertEqual(self.state['pending']['notes'], [])
        self.assertIn('выбор банкомата', self.state['response'])

    def test_deposit_ignores_spoken_amount_and_waits_for_device(self):
        self.say('deposit', amount=10000)
        command = self.state['outgoing']
        self.assertEqual(command['action'], 'start_deposit')
        self.assertIsNone(command['amount'])
        from atm_agent.atm.contracts import ATMCommand

        result = self.software.send(ATMCommand(**command))
        self.state.update(receive_result(self.state, result))
        self.assertEqual(self.state['result']['status'], 'pending')
        self.say('withdraw', amount=7000, denomination=1000)
        self.assertIsNone(self.state['outgoing'])
        actual = self.software.complete_deposit(command['request_id'], 12500)
        self.state.update(receive_result(self.state, actual))
        self.assertIn('12500', self.state['response'])
        self.assertIsNone(self.state['device_pending'])

    def test_result_must_match_request_and_contain_factual_value(self):
        self.say('balance')
        request_id = self.state['outgoing']['request_id']
        with self.assertRaises(OperationError):
            receive_result(self.state, ATMResult('another', 'completed', balance=12345))
        with self.assertRaises(OperationError):
            receive_result(self.state, ATMResult(request_id, 'completed'))
        updates = receive_result(self.state, ATMResult(request_id, 'completed', balance=12345))
        self.assertEqual(updates['response'], 'Баланс: 12345 ₽.')

    def test_failed_result_does_not_announce_success(self):
        self.say('withdraw', amount=7000, denomination=1000)
        self.say('confirm')
        request_id = self.state['outgoing']['request_id']
        updates = receive_result(self.state, ATMResult(request_id, 'failed', message='Нет купюр'))
        self.assertIn('отклонил', updates['response'])
        self.assertNotIn('выдаче', updates['response'])
