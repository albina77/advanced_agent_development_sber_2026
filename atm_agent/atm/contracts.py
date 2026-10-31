from dataclasses import dataclass
from typing import Literal, Protocol


class OperationError(ValueError):
    pass


@dataclass(frozen=True)
class ATMCommand:
    """Команда софту банкомата с идентификатором запроса и проверяемыми параметрами операции."""
    request_id: str
    action: Literal['get_balance', 'withdraw', 'start_deposit']
    amount: int | None = None
    denomination: int | None = None
    notes: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        """Нормализует состав купюр и проверяет соответствие аргументов действию и сумме."""
        try:
            object.__setattr__(self, 'notes', tuple((value, count) for value, count in self.notes))
        except (TypeError, ValueError) as exc:
            raise OperationError('Некорректный состав купюр.') from exc
        if not self.request_id:
            raise OperationError('Нужен идентификатор команды.')
        if self.action not in ('get_balance', 'withdraw', 'start_deposit'):
            raise OperationError('Неизвестная команда.')
        if self.action != 'withdraw':
            if self.amount is not None or self.denomination is not None or self.notes:
                raise OperationError('Сумма и купюры передаются только для выдачи.')
            return
        positive_int(self.amount)
        if self.denomination is not None:
            positive_int(self.denomination)
            if self.notes or self.amount % self.denomination:
                raise OperationError('Некорректный состав купюр.')
        if self.notes:
            if len(dict(self.notes)) != len(self.notes):
                raise OperationError('Номиналы не должны повторяться.')
            for value, count in self.notes:
                positive_int(value)
                positive_int(count)
            if sum(value * count for value, count in self.notes) != self.amount:
                raise OperationError('Сумма купюр не совпадает с суммой выдачи.')


def positive_int(value: int) -> None:
    if type(value) is not int or value <= 0:
        raise OperationError('Нужно положительное целое число.')


@dataclass(frozen=True)
class ATMResult:
    request_id: str
    status: Literal['completed', 'pending', 'failed']
    balance: int | None = None
    amount: int | None = None
    message: str = ''


class ATMSoftware(Protocol):
    """Граница интеграции с софтом банкомата; результат сообщает внешний софт."""

    def send(self, command: ATMCommand) -> ATMResult: ...
