import unittest

from atm_agent.atm.contracts import ATMCommand, ATMResult, OperationError
from atm_agent.atm.mock import DemoATMSoftware, MockATMSoftware


class ATMTest(unittest.TestCase):
    def test_demo_withdrawal_changes_balance_only_once(self):
        software = DemoATMSoftware()
        command = ATMCommand('withdraw', 'withdraw', 7000)
        first = software.send(command)
        self.assertEqual(software.send(command), first)
        balance = software.send(ATMCommand('balance', 'get_balance'))
        self.assertEqual(balance.balance, 43000)
        with self.assertRaises(OperationError):
            software.send(ATMCommand('withdraw', 'withdraw', 9000))
        self.assertEqual(software.balance, 43000)

    def test_demo_insufficient_funds_preserve_balance(self):
        software = DemoATMSoftware(initial_balance=3000)
        command = ATMCommand('withdraw', 'withdraw', 4000)
        self.assertEqual(software.send(command).status, 'failed')
        self.assertEqual(software.send(command).status, 'failed')
        self.assertEqual(software.balance, 3000)
        self.assertEqual(software.send(ATMCommand('empty', 'withdraw', 3000)).status, 'completed')
        self.assertEqual(software.balance, 0)

    def test_demo_deposit_credits_device_amount_only_once(self):
        software = DemoATMSoftware()
        software.send(ATMCommand('deposit', 'start_deposit'))
        self.assertEqual(software.balance, 50000)
        with self.assertRaises(OperationError):
            software.complete_deposit('deposit', -1)
        first = software.complete_deposit('deposit', 12500)
        self.assertEqual(software.complete_deposit('deposit', 12500), first)
        self.assertEqual(software.balance, 62500)
        with self.assertRaises(OperationError):
            software.complete_deposit('deposit', 13000)
        self.assertEqual(software.send(ATMCommand('balance', 'get_balance')).balance, 62500)

    def test_demo_failed_withdrawal_does_not_debit(self):
        software = DemoATMSoftware({'withdraw': ATMResult('withdraw', 'failed', message='Нет купюр')})
        software.send(ATMCommand('withdraw', 'withdraw', 7000))
        self.assertEqual(software.balance, 50000)

    def test_commands_forward_parameters(self):
        software = MockATMSoftware({'w': ATMResult('w', 'completed', amount=7000)})
        command = ATMCommand('w', 'withdraw', 7000, notes=((5000, 1), (1000, 2)))
        self.assertEqual(software.send(command).amount, 7000)
        self.assertEqual(software.commands, [command])

    def test_deposit_amount_comes_from_device(self):
        software = MockATMSoftware({'d': ATMResult('d', 'pending')})
        result = software.send(ATMCommand('d', 'start_deposit'))
        self.assertEqual(result.status, 'pending')
        self.assertIsNone(result.amount)
        result = software.complete_deposit('d', 12500)
        self.assertEqual(result.amount, 12500)
        self.assertEqual(result.status, 'completed')
        self.assertEqual(software.complete_deposit('d', 12500), result)
        with self.assertRaises(OperationError):
            software.complete_deposit('d', 13000)

    def test_rejected_withdrawal_is_not_success(self):
        software = MockATMSoftware({'w': ATMResult('w', 'failed', message='Нет купюр')})
        self.assertEqual(software.send(ATMCommand('w', 'withdraw', 7000)).status, 'failed')

    def test_balance_is_returned_by_software(self):
        software = MockATMSoftware({'b': ATMResult('b', 'completed', balance=12345)})
        self.assertEqual(software.send(ATMCommand('b', 'get_balance')).balance, 12345)

    def test_repeat_and_id_conflict(self):
        software = MockATMSoftware({'w': ATMResult('w', 'completed', amount=7000)})
        command = ATMCommand('w', 'withdraw', 7000)
        self.assertEqual(software.send(command), software.send(command))
        self.assertEqual(len(software.commands), 1)
        with self.assertRaises(OperationError):
            software.send(ATMCommand('w', 'withdraw', 5000))

    def test_invalid_commands(self):
        for kwargs in [dict(action='start_deposit', amount=1000),
                       dict(action='get_balance', amount=1000),
                       dict(action='withdraw', amount=True),
                       dict(action='withdraw', amount=-1),
                       dict(action='withdraw', amount=750, denomination=1000),
                       dict(action='withdraw', amount=7000, notes=((5000, 1),))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(OperationError):
                ATMCommand(request_id='x', **kwargs)


if __name__ == '__main__':
    unittest.main()
