from dataclasses import asdict, dataclass
from typing import Literal
from uuid import uuid4

from typing_extensions import TypedDict

from atm_agent.atm.contracts import ATMCommand, ATMResult, OperationError, positive_int
from atm_agent.understanding.input_gate import Relevance

Intent = Literal[
    'withdraw', 'details', 'confirm', 'reject', 'cancel', 'balance',
    'deposit', 'product', 'resume', 'pause', 'greeting', 'other', 'preference', 'memory_recall', 'forget_memory', 'capabilities',
]


@dataclass(frozen=True)
class Turn:
    relevance: Relevance
    intent: Intent = 'other'
    amount: int | None = None
    denomination: int | None = None
    notes: tuple[tuple[int, int], ...] | None = None
    atm_choice: bool = False
    preference_action: Literal['set', 'clear', 'keep'] = 'set'


class DialogueState(TypedDict, total=False):
    """Состояние сессии: параметры операции, подтверждения, история, память и диагностика вызовов."""
    text: str
    extracted: dict
    usage: dict
    slots: dict
    pending: dict | None
    confirmation_active: bool
    outgoing: dict | None
    device_pending: dict | None
    result: dict | None
    response: str | None
    history: list[dict]
    error: str | None
    citations: list[dict]
    retrieval_trace: list[dict]
    rag_tool_calls: list[dict]
    rag_calls: list[dict]
    rag_supported: bool
    rag_trace_file: str | None
    client_id: str
    memory_choice: dict | None
    memory_available: bool
    memory_error: str | None
    memory_event: dict | None
    memory_offer_invalidated: bool
    cash_offer: dict | None
    cash_offer_active: bool
    cash_offer_declined: bool
    preference_hint_shown: bool
    erase_memory_requested: bool
    memory_matches: list[dict]
    memory_index_status: dict
    deletion_report: dict
    summary: str
    summary_citations: list[str]
    summary_revision: int
    folded_messages: int
    summary_calls: list[dict]
    model_calls: list[dict]
    context_stats: dict
    context_error: str | None
    budget_log: list[dict]
    turn_usage: dict
    session_usage: dict
    turn_no: int


def confirmation_question(command: dict) -> str:
    if command['notes']:
        notes = ', '.join(f'{count} × {value} ₽' for value, count in command['notes'])
    elif command['denomination'] is not None:
        notes = f"купюрами по {command['denomination']} ₽"
    else:
        notes = 'купюрами на выбор банкомата'
    return f"Выдать {command['amount']} ₽, {notes}?"


def cash_offer_question(choice: dict) -> str:
    description = f"купюры по {choice['denomination']} ₽" if choice['denomination'] else 'выбор купюр банкоматом'
    if choice['kind'] == 'cash_preference':
        return f'Вы сохраняли предпочтение: {description}. Использовать его для этого снятия?'
    return f'В прошлый раз вы выбрали {description}. Выбрать так же для этого снятия?'


def cash_choice_question(state: DialogueState) -> dict:
    from atm_agent.memory.client import PREFERENCE_HINT

    response = 'Какими купюрами выдать? Можно на выбор банкомата.'
    updates = {}
    if state.get('memory_available') and not state.get('preference_hint_shown'):
        response += ' ' + PREFERENCE_HINT
        updates['preference_hint_shown'] = True
    return {**updates, 'response': response}


def command_state(command: ATMCommand) -> dict:
    return {**asdict(command), 'notes': [list(note) for note in command.notes]}


def withdrawal_changed(command: dict, turn: Turn) -> bool:
    """Отличает правку параметров от копирования моделью уже подтверждаемых значений."""
    for name in ('amount', 'denomination'):
        value = getattr(turn, name)
        if value is not None and (type(value) is not int or value != command[name]):
            return True
    if turn.notes is not None:
        if sorted(tuple(note) for note in turn.notes) != sorted(tuple(note) for note in command['notes']):
            return True
    if turn.atm_choice and (command['denomination'] is not None or command['notes']):
        return True
    selections = sum((turn.denomination is not None, bool(turn.notes), turn.atm_choice))
    return selections > 1


def handle_turn(state: DialogueState, turn: Turn) -> dict:
    """Обновляет диалог и предлагает команду; внешнего софта здесь не вызывает."""
    updates = {'response': None, 'outgoing': None, 'result': None, 'error': None}
    if turn.relevance != Relevance.ADDRESSED:
        return updates
    if state.get('device_pending'):
        return {**updates, 'response': 'Ожидаю результат текущей операции от банкомата.'}

    intent = turn.intent
    if intent == 'pause':
        return {**updates, 'confirmation_active': False, 'cash_offer_active': False,
                'response': 'Параметры сохранены. Когда будете готовы, скажите: «Продолжим снятие».'}
    if state.get('memory_offer_invalidated') and intent in ('confirm', 'resume'):
        return {**updates, 'confirmation_active': False,
                'response': 'Сохранённый вариант больше недоступен. Укажите купюры для этого снятия.'}
    if intent == 'cancel':
        return {**updates, 'slots': {}, 'pending': None, 'confirmation_active': False,
                'cash_offer': None, 'cash_offer_active': False, 'cash_offer_declined': False,
                'response': 'Операция отменена.'}
    if intent == 'product':
        return {**updates, 'confirmation_active': False, 'cash_offer_active': False,
                'response': 'Консультации по документам пока недоступны.'}
    if intent == 'capabilities':
        return {**updates, 'confirmation_active': False, 'cash_offer_active': False,
                'response': 'Да, здесь можно снять или внести наличные и посмотреть баланс. '
                            'Чтобы начать внесение, скажите: «Хочу внести наличные». '
                            'Сумму внесения определит купюроприёмник.'}
    if intent in ('balance', 'deposit'):
        action = 'get_balance' if intent == 'balance' else 'start_deposit'
        return {**updates, 'confirmation_active': False, 'cash_offer_active': False,
                'outgoing': command_state(ATMCommand(uuid4().hex, action))}
    if intent == 'greeting':
        return {**updates, 'confirmation_active': False, 'cash_offer_active': False,
                'response': 'Здравствуйте! Могу помочь снять или внести наличные, показать баланс '
                            'и проконсультировать по СберПрайм, СберСпасибо и СберКарте. Что вас интересует?'}

    has_details = any((turn.amount is not None, turn.denomination is not None,
                       turn.notes is not None, turn.atm_choice))
    pending = state.get('pending')
    offer = state.get('cash_offer')
    unchanged_offer = offer is not None and not withdrawal_changed({
        'amount': state.get('slots', {}).get('amount'),
        'denomination': offer['denomination'], 'notes': [],
    }, turn)
    if offer and intent == 'reject' and has_details:
        intent = 'details'
    if offer and intent in ('confirm', 'resume') and (not has_details or unchanged_offer):
        if intent == 'confirm' and state.get('cash_offer_active'):
            return handle_turn({**state, 'cash_offer': None, 'cash_offer_active': False},
                               Turn(Relevance.ADDRESSED, 'details', denomination=offer['denomination'],
                                    atm_choice=offer['atm_choice'])) | {'cash_offer': None, 'cash_offer_active': False}
        return {**updates, 'cash_offer_active': True, 'response': cash_offer_question(offer)}
    if offer and intent == 'reject':
        return {**updates, 'cash_offer': None, 'cash_offer_active': False, 'cash_offer_declined': True,
                'confirmation_active': False, **cash_choice_question(state)}
    unchanged_confirmation = pending is not None and not withdrawal_changed(pending, turn)
    if intent == 'confirm' and (not has_details or unchanged_confirmation):
        if pending and state.get('confirmation_active'):
            return {**updates, 'outgoing': pending, 'pending': None, 'slots': {},
                    'confirmation_active': False, 'cash_offer_declined': False}
        if pending:
            return {**updates, 'response': confirmation_question(pending),
                    'confirmation_active': True}
        return {**updates, 'response': 'Сначала укажите, какую операцию хотите выполнить.'}
    if intent == 'reject':
        return {**updates, 'pending': None, 'confirmation_active': False,
                'response': 'Выдачу не отправляю. Уточните сумму или купюры либо отмените операцию.'}
    if intent == 'resume' and pending and not has_details:
        return {**updates, 'response': confirmation_question(pending),
                'confirmation_active': True}
    if intent not in ('withdraw', 'details', 'resume', 'confirm'):
        return {**updates, 'confirmation_active': False, 'cash_offer_active': False,
                'response': 'Могу помочь со снятием, внесением наличных и балансом, а также '
                            'проконсультировать по СберПрайм, СберСпасибо и СберКарте. '
                            'Например, спросите, какие сервисы входят в СберПрайм.'}
    if intent != 'withdraw' and not state.get('slots') and not pending:
        return {**updates, 'response': 'Какую операцию хотите выполнить?'}

    slots = dict(state.get('slots', {}))
    slots['action'] = 'withdraw'
    # Сначала отменяем старое предложение, в том числе при некорректной правке.
    updates.update(slots=slots, pending=None, confirmation_active=False, cash_offer=None, cash_offer_active=False)
    try:
        if turn.amount is not None:
            slots.pop('amount', None)
            positive_int(turn.amount)
            slots['amount'] = turn.amount
        selections = sum((turn.denomination is not None, turn.notes is not None, turn.atm_choice))
        if selections > 1:
            raise OperationError('Укажите один вариант выдачи купюр.')
        if turn.denomination is not None:
            slots.update(denomination=None, notes=[], cash_choice=False)
            positive_int(turn.denomination)
            slots.update(denomination=turn.denomination, notes=[], cash_choice=True)
        elif turn.notes is not None:
            if not turn.notes:
                raise OperationError('Укажите состав купюр или выбор банкомата.')
            slots.update(denomination=None, notes=[list(note) for note in turn.notes], cash_choice=True)
        elif turn.atm_choice:
            slots.update(denomination=None, notes=[], cash_choice=True)
        if 'amount' not in slots:
            return {**updates, 'response': 'Какую сумму хотите снять?'}
        if not slots.get('cash_choice'):
            choice = state.get('memory_choice') if not state.get('cash_offer_declined') else None
            if choice and (choice['denomination'] is None or slots['amount'] % choice['denomination'] == 0):
                return {**updates, 'cash_offer': choice, 'cash_offer_active': True,
                        'response': cash_offer_question(choice)}
            return {**updates, **cash_choice_question(state)}
        command = ATMCommand(uuid4().hex, 'withdraw', slots['amount'],
                             slots.get('denomination'), tuple(slots.get('notes', ())))
    except OperationError as exc:
        return {**updates, 'response': str(exc)}
    pending = command_state(command)
    return {**updates, 'pending': pending, 'confirmation_active': True,
            'response': confirmation_question(pending)}


def receive_result(state: DialogueState, result: ATMResult) -> dict:
    """Переводит ответ софта в состояние диалога, проверяя связь с командой."""
    command = state.get('outgoing') or state.get('device_pending')
    if not command or result.request_id != command['request_id']:
        raise OperationError('Ответ софта не относится к ожидаемой команде.')
    updates = {'outgoing': None, 'result': asdict(result), 'device_pending': None, 'error': None}
    if result.status == 'pending':
        return {**updates, 'device_pending': command,
                'response': 'Внесите купюры в купюроприёмник.' if command['action'] == 'start_deposit'
                else 'Банкомат выполняет операцию.'}
    if result.status == 'failed':
        return {**updates, 'response': f'Банкомат отклонил операцию: {result.message}'}
    if result.status != 'completed':
        raise OperationError('Неизвестный статус операции от банкомата.')
    if command['action'] == 'get_balance':
        if type(result.balance) is not int or result.balance < 0:
            raise OperationError('Софт не сообщил корректный баланс.')
        return {**updates, 'response': f'Баланс: {result.balance} ₽.'}
    if type(result.amount) is not int or result.amount < 0:
        raise OperationError('Софт не сообщил фактическую сумму операции.')
    if command['action'] == 'withdraw':
        return {**updates, 'response': f'Банкомат сообщил о выдаче {result.amount} ₽.'}
    return {**updates, 'response': f'Зачислено {result.amount} ₽ по данным банкомата.'}
