import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Literal

from langgraph.store.base import BaseStore
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from atm_agent.atm.contracts import OperationError, positive_int

PREFERENCE_HINT = 'Чтобы сохранить предпочтение, скажите: «Запомни: обычно выдавай тысячными».'


class MemoryRecord(BaseModel):
    """Сведения из явной просьбы клиента или успешного ответа устройства."""

    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[1] = 1
    kind: Literal['cash_preference', 'withdrawal_episode']
    source: Literal['user_explicit', 'atm_completed']
    denomination: StrictInt | None = None
    atm_choice: bool = False
    amount: StrictInt | None = None
    notes: list[tuple[StrictInt, StrictInt]] = Field(default_factory=list)
    request_id: str | None = None
    created_at: datetime
    expires_at: datetime

    @model_validator(mode='after')
    def check_record(self):
        """Проверяет сроки и допустимые поля для явного предпочтения или подтверждённого эпизода."""
        if any(value.tzinfo is None for value in (self.created_at, self.expires_at)):
            raise ValueError('Время записи должно содержать часовой пояс.')
        if self.expires_at <= self.created_at:
            raise ValueError('Срок действия должен быть положительным.')
        if self.denomination is not None:
            positive_int(self.denomination)
        if self.kind == 'cash_preference':
            if (self.source != 'user_explicit' or self.amount is not None or self.notes or self.request_id
                    or sum((self.denomination is not None, self.atm_choice)) != 1):
                raise ValueError('Предпочтение задаёт один способ выбора купюр, без суммы.')
        else:
            if self.source != 'atm_completed' or not self.request_id:
                raise ValueError('Эпизоду нужен успешный результат устройства.')
            positive_int(self.amount)
            if sum((self.denomination is not None, bool(self.notes), self.atm_choice)) != 1:
                raise ValueError('У эпизода должен быть один способ выбора купюр.')
        return self


class ClientMemory:
    """Отделяет профиль от эпизодов; явное предпочтение важнее прошлой выдачи."""

    def __init__(self, store: BaseStore, preference_days=180, episode_days=30,
                 clock: Callable[[], datetime] | None = None):
        positive_int(preference_days)
        positive_int(episode_days)
        self.store = store
        self.preference_days, self.episode_days = preference_days, episode_days
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def namespace(client_id, role):
        if not isinstance(client_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', client_id):
            raise ValueError('Нужен идентификатор клиента от считывателя карты.')
        return ('clients', client_id, role)

    def choice(self, client_id):
        """Выбирает действующее явное предпочтение, а при его отсутствии — последний подходящий эпизод."""
        now = self.clock()
        item = self.store.get(self.namespace(client_id, 'profile'), 'cash_preference')
        if item:
            record = MemoryRecord.model_validate(item.value)
            if record.created_at <= now < record.expires_at:
                return self._choice(record)
        episodes = self.store.search(self.namespace(client_id, 'episodes'), limit=10000)
        records = [MemoryRecord.model_validate(item.value) for item in episodes]
        valid = [record for record in records if record.created_at <= now < record.expires_at]
        if valid:
            latest = max(valid, key=lambda record: (record.created_at, record.request_id))
            # Точный состав относится к прежней сумме и не переносится на новую.
            if not latest.notes:
                return self._choice(latest)
        return None

    @staticmethod
    def _choice(record):
        return {'kind': record.kind, 'denomination': record.denomination, 'atm_choice': record.atm_choice,
                'created_at': record.created_at.isoformat(), 'expires_at': record.expires_at.isoformat()}

    def set_preference(self, client_id, denomination=None, atm_choice=False):
        now = self.clock()
        try:
            record = MemoryRecord(kind='cash_preference', source='user_explicit',
                                  denomination=denomination, atm_choice=atm_choice,
                                  created_at=now, expires_at=now + timedelta(days=self.preference_days))
        except ValueError as exc:
            raise OperationError('Укажите один номинал или выбор банкомата для будущих снятий.') from exc
        self.store.put(self.namespace(client_id, 'profile'), 'cash_preference', record.model_dump(mode='json'))

    def clear_preference(self, client_id):
        self.store.delete(self.namespace(client_id, 'profile'), 'cash_preference')

    def record_withdrawal(self, client_id, command, result):
        """Сохраняет эпизод только при совпадении успешного результата устройства с командой."""
        if (command['action'] != 'withdraw' or result['status'] != 'completed'
                or result['request_id'] != command['request_id'] or result['amount'] != command['amount']):
            return
        namespace = self.namespace(client_id, 'episodes')
        if self.store.get(namespace, command['request_id']):
            return
        now = self.clock()
        record = MemoryRecord(kind='withdrawal_episode', source='atm_completed',
                              amount=result['amount'], denomination=command['denomination'],
                              notes=command['notes'], atm_choice=not command['denomination'] and not command['notes'],
                              request_id=command['request_id'], created_at=now,
                              expires_at=now + timedelta(days=self.episode_days))
        self.store.put_episode(client_id, command['request_id'], record.model_dump(mode='json'))

    def recall(self, client_id, query):
        """Формирует ответ из найденных записей и сообщает о ещё не подготовленной части памяти."""
        prefix = self.namespace(client_id, 'profile')[:2]
        status = self.store.index_status(client_id)
        if self.store.embeddings is None:
            return {'response': 'Поиск по памяти сейчас недоступен.', 'matches': [], 'index_status': status}
        items = self.store.search(prefix, query=query, limit=5)
        matches, lines = [], []
        for item in sorted(items, key=lambda item: item.value['kind'] != 'cash_preference'):
            value = {key: value for key, value in item.value.items() if not key.startswith('_')}
            record = MemoryRecord.model_validate(value)
            if record.notes:
                choice = ', '.join(f'{count} × {denomination} ₽' for denomination, count in record.notes)
            else:
                choice = f'купюры по {record.denomination} ₽' if record.denomination else 'выбор купюр банкоматом'
            if record.kind == 'cash_preference':
                lines.append(f'Сохранённое предпочтение: {choice}.')
            else:
                lines.append(f'Успешная выдача {record.created_at:%d.%m.%Y}: {record.amount} ₽, {choice}.')
            matches.append({'kind': record.kind, 'key': item.key, 'score': item.score,
                            **item.value['_retrieval']})
        response = '\n'.join(lines) if lines else 'По этому вопросу подходящих записей памяти не найдено.'
        if status['pending']:
            response += ' Часть записей ещё не подготовлена для поиска.'
        return {'response': response, 'matches': matches, 'index_status': status}

    def forget(self, client_id):
        self.namespace(client_id, 'profile')
        return self.store.forget(client_id)
