import hashlib
import json
import os
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.ai import add_usage
from langchain_core.utils.function_calling import convert_to_openai_tool

from atm_agent.config.paths import PROMPTS


class ContextBudgetError(RuntimeError):
    def __init__(self, reason, calls=None, preflight=None):
        self.reason, self.calls, self.preflight = reason, list(calls or []), preflight
        super().__init__(reason)


def total_usage(records):
    usage = None
    for record in records:
        if record.get('usage'):
            usage = add_usage(usage, record['usage'])
    return usage or {}


def history_turns(history):
    """Не разрывает пары клиент/агент и ответы на события устройства."""
    groups = []
    for message in history:
        if message['role'] == 'user' or not groups:
            groups.append([])
        groups[-1].append(message)
    return groups


CITATION = re.compile(r'\[([A-Za-z0-9_@.\-]+),\s*с\.\s*(\d+)\]')


def citation_refs(previous, messages):
    refs = list(previous)
    for message in messages:
        for source, page in CITATION.findall(message['content']):
            if source not in ('source_id', 'ID'):
                label = f'[{source}, с. {int(page)}]'
                if label in refs:
                    refs.remove(label)
                refs.append(label)
    return refs[-12:]


class ContextBudget:
    """Проверка токенизатором до вызова и учёт фактического usage после него.

    Считается сериализованное содержимое, включая схемы инструментов.
    Запас покрывает различия с внутренним форматом чата; это preflight,
    а не фактическая стоимость запроса.
    """

    def __init__(self, model, input_tokens=4096, rag_tokens=12000, summary_tokens=384,
                 recent_turns=4, overhead=256, counter=None):
        if min(input_tokens, rag_tokens, summary_tokens, recent_turns) <= 0 or overhead < 0:
            raise ValueError('Бюджеты контекста должны быть положительными.')
        self.model = model
        self.input_tokens, self.rag_tokens = input_tokens, rag_tokens
        self.summary_tokens, self.recent_turns, self.overhead = summary_tokens, recent_turns, overhead
        self.counter = counter
        self.cache = {}
        self.summary_prompt = (PROMPTS / 'summary_prompt.txt').read_text()

    @classmethod
    def from_env(cls, model):
        return cls(model, input_tokens=int(os.getenv('CONTEXT_INPUT_TOKENS', '4096')),
                   rag_tokens=int(os.getenv('RAG_INPUT_TOKENS', '12000')),
                   summary_tokens=int(os.getenv('CONTEXT_SUMMARY_TOKENS', '384')),
                   recent_turns=int(os.getenv('CONTEXT_RECENT_TURNS', '4')))

    def count(self, messages, schemas=()):
        """Считает токены сообщений и схем через API токенизатора; кэш хранит только хеши и числа."""
        payload = {'messages': [message.model_dump(exclude_none=True) for message in messages],
                   'tools': [convert_to_openai_tool(schema) for schema in schemas]}
        text = json.dumps(payload, ensure_ascii=False, default=str)
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in self.cache:
            count = self.counter(text) if self.counter else self.model.tokens_count([text], model=self.model.model)[0].tokens
            if type(count) is not int or count < 0:
                raise ValueError('Токенизатор не сообщил корректный размер контекста.')
            if len(self.cache) >= 128:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = count
        return self.cache[key]

    def check(self, stage, messages, schemas=(), limit=None):
        limit = self.input_tokens if limit is None else limit
        count = self.count(messages, schemas)
        preflight = {'stage': stage, 'counted_tokens': count, 'overhead_reserve': self.overhead,
                     'input_budget': limit, 'max_output_tokens': self.summary_tokens if stage == 'summarize'
                     else getattr(self.model, 'max_tokens', None),
                     'counter': 'test_counter' if self.counter else 'gigachat_tokens_count'}
        if count + self.overhead > limit:
            raise ContextBudgetError('input_budget_exceeded', preflight=preflight)
        return preflight

    @staticmethod
    def record(stage, raw, preflight, elapsed):
        usage = getattr(raw, 'usage_metadata', None) or {}
        actual = usage.get('input_tokens')
        return {'stage': stage, 'usage': usage, 'preflight': preflight, 'latency_ms': elapsed * 1000,
                'within_budget': actual <= preflight['input_budget'] if actual is not None else None}

    def prepare(self, state, messages_fn, schema):
        calls = []
        try:
            return self._prepare(state, messages_fn, schema, calls)
        except Exception as exc:
            if isinstance(exc, ContextBudgetError):
                exc.calls = calls
                raise
            raise ContextBudgetError(f'context_preparation_failed:{type(exc).__name__}', calls) from exc

    def _prepare(self, state, messages_fn, schema, calls):
        """Сворачивает старые ходы в сводку, сохраняя последние реплики и ссылки отдельно."""
        history = state.get('history', [])
        groups = history_turns(history)
        summary = state.get('summary', '')
        refs = state.get('summary_citations', [])
        before = self.count(messages_fn(state['text'], state), [schema])
        updates = {'summary_calls': [], 'summary': summary, 'summary_citations': refs, 'context_error': None}
        if before + self.overhead <= self.input_tokens and len(groups) <= self.recent_turns * 2:
            return {**updates, 'context_stats': {'action': 'keep', 'preflight_before': before,
                    'preflight_after': before, 'window_messages': len(history), 'folded_messages': 0,
                    'input_budget': self.input_tokens}}
        if len(groups) < 2:
            raise ContextBudgetError('mandatory_context_exceeds_budget')
        tail, dropped = None, None
        for keep in range(min(self.recent_turns, len(groups) - 1), 0, -1):
            candidate = [message for group in groups[-keep:] for message in group]
            folded = [message for group in groups[:-keep] for message in group]
            preview = {**state, 'history': candidate, 'summary': '',
                       'summary_citations': citation_refs(refs, folded)}
            size = self.count(messages_fn(state['text'], preview), [schema])
            if size + self.overhead + self.summary_tokens + 64 <= self.input_tokens:
                tail = candidate
                dropped = groups[:-keep]
                break
        if tail is None:
            raise ContextBudgetError('mandatory_context_exceeds_budget')
        remaining = list(dropped)
        refs = citation_refs(refs, [message for group in dropped for message in group])
        while remaining:
            size = len(remaining)
            while size:
                batch = [message for group in remaining[:size] for message in group]
                messages = [SystemMessage(self.summary_prompt), HumanMessage(json.dumps({
                    'previous_summary': summary, 'history_to_fold': batch,
                }, ensure_ascii=False))]
                try:
                    preflight = self.check('summarize', messages)
                    break
                except ContextBudgetError:
                    size //= 2
            if not size:
                raise ContextBudgetError('summary_chunk_exceeds_budget')
            start = time.perf_counter()
            response = self.model.bind(max_tokens=self.summary_tokens).invoke(messages)
            record = self.record('summarize', response, preflight, time.perf_counter() - start)
            calls.append(record)
            if record['within_budget'] is False:
                raise ContextBudgetError('actual_summary_budget_exceeded')
            text = response.content
            if (not isinstance(text, str) or not text.strip() or response.tool_calls
                    or response.response_metadata.get('finish_reason') in ('length', 'max_tokens')):
                raise ContextBudgetError('invalid_summary')
            if len(text) > 2400:
                raise ContextBudgetError('summary_too_long')
            summary = re.sub(r'\[source_id[^\]]*\]', '', CITATION.sub('', text)).strip()
            remaining = remaining[size:]
        prepared = {**state, 'history': tail, 'summary': summary, 'summary_citations': refs}
        after = self.check('extract', messages_fn(state['text'], prepared), [schema])
        return {**updates, 'summary': summary, 'summary_citations': refs, 'history': tail, 'summary_calls': calls,
                'summary_revision': state.get('summary_revision', 0) + len(calls),
                'folded_messages': state.get('folded_messages', 0) + len(history) - len(tail),
                'context_stats': {'action': 'summarize', 'preflight_before': before,
                    'preflight_after': after['counted_tokens'], 'window_messages': len(tail),
                    'folded_messages': len(history) - len(tail), 'input_budget': self.input_tokens}}
