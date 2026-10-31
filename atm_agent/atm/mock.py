from atm_agent.atm.contracts import ATMCommand, ATMResult, OperationError


class MockATMSoftware:
    """Заглушка транспорта с заранее заданными ответами внешнего софта."""

    def __init__(self, responses: dict[str, ATMResult]):
        self.responses = dict(responses)
        self.commands: list[ATMCommand] = []
        self._received: dict[str, ATMCommand] = {}

    def send(self, command: ATMCommand) -> ATMResult:
        previous = self._received.get(command.request_id)
        if previous is not None and previous != command:
            raise OperationError('Идентификатор уже использован другой командой.')
        result = self.responses.get(command.request_id)
        if result is None or result.request_id != command.request_id:
            raise OperationError('Для команды не задан ответ софта банкомата.')
        if previous is None:
            self._received[command.request_id] = command
            self.commands.append(command)
        return result
    def complete_deposit(self, request_id: str, accepted_amount: int) -> ATMResult:
        """Имитирует событие купюроприёмника, а не сообщение клиента."""
        command = self._received.get(request_id)
        if command is None or command.action != 'start_deposit':
            raise OperationError('Нет такой операции внесения.')
        if type(accepted_amount) is not int or accepted_amount < 0:
            raise OperationError('Некорректная сумма от купюроприёмника.')
        previous = self.responses[request_id]
        if previous.status != 'pending':
            if previous.status == 'completed' and previous.amount == accepted_amount:
                return previous
            raise OperationError('Операция уже завершена.')
        result = ATMResult(request_id, 'completed', amount=accepted_amount)
        self.responses[request_id] = result
        return result


class DemoATMSoftware(MockATMSoftware):
    """Симулятор одного счёта: баланс меняется после успешных операций в пределах процесса."""

    def __init__(self, responses=None, initial_balance=50000):
        if type(initial_balance) is not int or initial_balance < 0:
            raise OperationError('Начальный баланс должен быть неотрицательным целым числом.')
        super().__init__(responses or {})
        self.balance = initial_balance

    def send(self, command: ATMCommand) -> ATMResult:
        """Выполняет новую команду; повтор возвращает прежний результат без повторного списания."""
        first_request = command.request_id not in self._received
        if command.request_id not in self.responses:
            if command.action == 'get_balance':
                result = ATMResult(command.request_id, 'completed', balance=self.balance)
            elif command.action == 'withdraw':
                if command.amount > self.balance:
                    result = ATMResult(command.request_id, 'failed', message='Недостаточно средств на счёте.')
                else:
                    result = ATMResult(command.request_id, 'completed', amount=command.amount)
            else:
                result = ATMResult(command.request_id, 'pending')
            self.responses[command.request_id] = result
        result = super().send(command)
        if first_request and command.action == 'withdraw' and result.status == 'completed':
            self.balance -= command.amount
        return result

    def complete_deposit(self, request_id: str, accepted_amount: int) -> ATMResult:
        """Зачисляет сумму купюроприёмника один раз, только при завершении ожидающего внесения."""
        previous = self.responses.get(request_id)
        result = super().complete_deposit(request_id, accepted_amount)
        if previous.status == 'pending':
            self.balance += result.amount
        return result
