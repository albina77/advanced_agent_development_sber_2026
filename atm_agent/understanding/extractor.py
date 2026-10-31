import json
import re
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from atm_agent.config.paths import PROMPTS
from atm_agent.dialogue.state import Turn
from atm_agent.understanding.input_gate import Relevance


class Banknote(BaseModel):
    """Номинал и количество купюр в структурированном ответе модели."""
    model_config = ConfigDict(extra='forbid')
    denomination: StrictInt = Field(description='Номинал одной купюры в рублях')
    count: StrictInt = Field(description='Количество купюр этого номинала')


class ParsedTurn(BaseModel):
    """Адресованность и явно сказанные параметры текущей реплики."""

    model_config = ConfigDict(extra='forbid')
    relevance: Literal['addressed', 'background', 'uncertain']
    intent: Literal[
        'withdraw', 'details', 'confirm', 'reject', 'cancel', 'balance',
        'deposit', 'product', 'resume', 'pause', 'greeting', 'other', 'preference', 'memory_recall', 'forget_memory', 'capabilities',
    ]
    amount: StrictInt | None = Field(default=None, description='Явно названная сумма снятия, рубли')
    denomination: StrictInt | None = Field(default=None, description='Один желаемый номинал купюр текущего снятия')
    notes: list[Banknote] | None = Field(default=None, description='Точный состав разовой выдачи по номиналам')
    atm_choice: bool = Field(default=False, description='Клиент явно разрешил банкомату выбрать купюры текущего снятия')
    amount_quote: str | None = Field(default=None, description='Дословные слова текущей реплики, задающие сумму; иначе null')
    denomination_quote: str | None = Field(default=None, description='Дословные слова текущей реплики с выбранным номиналом; иначе null')
    notes_quote: str | None = Field(default=None, description='Дословные слова текущей реплики с составом купюр; иначе null')
    atm_choice_quote: str | None = Field(default=None, description='Дословное разрешение банкомату выбрать купюры в текущей реплике; иначе null')

    def to_turn(self, text: str | None = None) -> Turn:
        """При переданном тексте оставляет параметры только с дословной опорой в текущей реплике."""
        def grounded(quote, numeric=False):
            if text is None:
                return True
            if not quote:
                return False
            normalized = lambda value: ' '.join(value.casefold().split())
            return (bool(normalized(quote)) and normalized(quote) in normalized(text) and
                    (not numeric or re.search(r'\d|\b(?:нул|нол|одн|оди|дв|тр|четыр|пят|шест|сем|восем|дев|десят|сто|сот|тысяч|миллион)',
                                               quote.casefold()) is not None))

        return Turn(
            Relevance(self.relevance), self.intent,
            self.amount if grounded(self.amount_quote, True) else None,
            self.denomination if grounded(self.denomination_quote, True) else None,
            tuple((note.denomination, note.count) for note in self.notes)
            if self.notes and grounded(self.notes_quote, True) else None,
            self.atm_choice and grounded(self.atm_choice_quote),
        )


class ParsedPreference(BaseModel):
    """Явная просьба о будущем выборе купюр, без параметров текущего снятия."""

    model_config = ConfigDict(extra='forbid')
    action: Literal['set', 'clear', 'keep']
    denomination: StrictInt | None = Field(default=None, description='Номинал для будущих выдач, например 500')
    atm_choice: bool = Field(default=False, description='В будущем купюры выбирает банкомат')


class ExtractionError(RuntimeError):
    """Ошибка извлечения с причиной и уже известным расходом токенов."""
    def __init__(self, usage: dict | None = None, reason='extraction_failed'):
        super().__init__('Не удалось получить корректные параметры реплики.')
        self.usage = usage or {}
        self.reason = reason


class GigaChatExtractor:
    """Извлекает намерение и параметры через structured output с контролем контекста и расхода токенов."""
    schema = ParsedTurn

    def __init__(self, model, context=None):
        from atm_agent.context.budget import ContextBudget

        self.context = context or (ContextBudget.from_env(model) if hasattr(model, 'tokens_count') else None)
        self.last_calls = []
        self.structured = model.with_structured_output(ParsedTurn, include_raw=True)
        self.preference = model.with_structured_output(ParsedPreference, include_raw=True)
        self.prompt = (PROMPTS / 'extractor_prompt.txt').read_text()
        self.preference_prompt = (PROMPTS / 'preference_prompt.txt').read_text()

    def messages(self, text: str, state: dict):
        """Собирает контекст извлечения из разрешённых полей состояния без идентификатора клиента."""
        from langchain_core.messages import HumanMessage, SystemMessage

        context = {
            'slots': state.get('slots', {}),
            'pending': state.get('pending'),
            'confirmation_active': state.get('confirmation_active', False),
            'device_pending': state.get('device_pending'),
            'cash_offer': state.get('cash_offer'),
            'cash_offer_active': state.get('cash_offer_active', False),
            'summary': state.get('summary', ''),
            'summary_citations': state.get('summary_citations', []),
            'history': state.get('history', []) if self.context else state.get('history', [])[-12:],
        }
        return [
            SystemMessage(self.prompt),
            HumanMessage(json.dumps({'context': context, 'utterance': text}, ensure_ascii=False)),
        ]

    def _invoke(self, runnable, messages, schema, stage):
        """Проверяет бюджет, вызывает модель и учитывает фактический расход даже при превышении лимита."""
        from atm_agent.context.budget import (
            ContextBudget,
            ContextBudgetError,
            total_usage,
        )

        try:
            preflight = self.context.check(stage, messages, [schema]) if self.context else None
            start = time.perf_counter()
            output = runnable.invoke(messages)
        except Exception as exc:
            reason = 'context_budget_exceeded' if isinstance(exc, ContextBudgetError) else 'extraction_failed'
            raise ExtractionError(total_usage(self.last_calls), reason) from exc
        raw = output.get('raw')
        if preflight:
            record = ContextBudget.record(stage, raw, preflight, time.perf_counter() - start)
        else:
            record = {'stage': stage, 'usage': getattr(raw, 'usage_metadata', None) or {}}
        self.last_calls.append(record)
        if record.get('within_budget') is False:
            raise ExtractionError(total_usage(self.last_calls), 'actual_context_budget_exceeded')
        return output

    def extract(self, text: str, state: dict) -> tuple[Turn, dict]:
        """Извлекает текущую реплику; сводка остаётся контекстом, а не источником команды."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from atm_agent.context.budget import total_usage

        self.last_calls = []
        output = self._invoke(self.structured, self.messages(text, state), ParsedTurn, 'extract')
        usage = total_usage(self.last_calls)
        parsed = output.get('parsed')
        if output.get('parsing_error') or not isinstance(parsed, ParsedTurn):
            raise ExtractionError(usage)
        if parsed.relevance == 'addressed' and parsed.intent == 'preference':
            preference_output = self._invoke(self.preference, [
                SystemMessage(self.preference_prompt), HumanMessage(text),
            ], ParsedPreference, 'preference')
            preference = preference_output.get('parsed')
            usage = total_usage(self.last_calls)
            if preference_output.get('parsing_error') or not isinstance(preference, ParsedPreference):
                raise ExtractionError(usage)
            return Turn(Relevance.ADDRESSED, 'preference', denomination=preference.denomination,
                        atm_choice=preference.atm_choice, preference_action=preference.action), usage
        return parsed.to_turn(text), usage
