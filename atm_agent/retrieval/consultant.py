import argparse
import json
import os
import re
import time
from datetime import date, datetime, timezone
from typing import Literal
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from atm_agent.config.paths import PROMPTS
from atm_agent.config.settings import create_embeddings, create_model
from atm_agent.knowledge.corpus import INDEX
from atm_agent.knowledge.versions import parse_date
from atm_agent.retrieval.index import HybridIndex


class SearchInput(BaseModel):
    """Запрос инструмента поиска с датой и необязательными фильтрами документов."""
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1, max_length=1500, description='Вопрос для поиска по документам')
    as_of: str = Field(default='', description='Дата YYYY-MM-DD из вопроса; пусто означает сегодня')
    product: Literal['', 'sberprime', 'sberspasibo', 'sberkarta'] = ''
    document_id: str = Field(default='', description='Семейство документа из предоставленного каталога')
    source_id: str = Field(default='', description='Точный идентификатор редакции, если он указан клиентом')
    include_undated: bool = Field(default=False, description='Для общего ознакомления без даты; актуальность недатированных документов не подтверждена')


class Evidence(BaseModel):
    """Дословная цитата и идентификатор найденного фрагмента, подтверждающие утверждение."""
    model_config = ConfigDict(extra='forbid')
    chunk_id: str
    quote: str = Field(default='', description='Дословная опора; оставь пустой при выборе quote_id')
    quote_id: str = Field(default='', description='Идентификатор готовой цитаты из quote_options выбранного фрагмента')


class Claim(BaseModel):
    """Утверждение ответа с одной или несколькими опорными цитатами."""
    model_config = ConfigDict(extra='forbid')
    text: str = Field(min_length=1, description='Одно утверждение, подтверждённое цитатами')
    evidence: list[Evidence] = Field(min_length=1)


class GroundedAnswer(BaseModel):
    """Ответ состоит только из утверждений с проверяемыми опорами в найденных фрагментах."""

    model_config = ConfigDict(extra='forbid')
    supported: bool = Field(description='В найденных документах достаточно сведений для ответа?')
    claims: list[Claim] = Field(default_factory=list)
    clarification: str = Field(default='', description='Один уточняющий вопрос без утверждений о банковских условиях, если данных для ответа недостаточно')


class RejectedClaim(BaseModel):
    model_config = ConfigDict(extra='forbid')
    index: StrictInt = Field(ge=0)
    reason: str


class SupportReview(BaseModel):
    """Утверждения, для которых источник не подтверждает смысл, числа или область применения."""
    model_config = ConfigDict(extra='forbid')
    rejected: list[RejectedClaim] = Field(default_factory=list)


def normalize_quote(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def quote_options(text: str) -> dict[str, str]:
    """Делит нормализованный текст на короткие точные опоры без генерации новых слов."""
    words = normalize_quote(text).split()
    parts, current = [], []
    for word in words:
        if current and len(' '.join([*current, word])) > 180:
            parts.append(' '.join(current))
            current = []
        current.append(word)
    if current:
        parts.append(' '.join(current))
    return {f'q{index}': part for index, part in enumerate(parts) if len(part) >= 8}


def overview_question(question: str) -> bool:
    """Разрешает недатированный источник только для общего знакомства без запроса актуальных условий."""
    text = question.casefold()
    return (bool(re.search(r'что такое|расскажи (?:мне )?(?:о|про)|что за (?:карта|продукт)', text))
            and not re.search(r'\d|сегодня|сейчас|актуал|действу|тариф|стоим|стоит|лимит|комисс|процент|цен|дат', text))


def overview_options(text: str) -> dict[str, str]:
    """Выбирает описательные строки недатированного документа без чисел и тарифных условий."""
    lines = [normalize_quote(line) for line in text.splitlines()]
    allowed = [line for line in lines if len(line) >= 8
               and re.search(r'карт|подписк|продукт|сервис', line.casefold())
               and not re.search(r'\d|₽|рубл|бесплат|стоим|комисс|лимит|процент|скид|при |услов|возраст', line.casefold())]
    return {f'q{index}': line for index, line in enumerate(allowed)}


def undated_conditions(claim: Claim, chunks: dict, undated: set[str]) -> bool:
    """Блокирует числовые и тарифные условия из недатированных источников в ознакомительном режиме."""
    return (any(chunks[item.chunk_id]['source_id'] in undated for item in claim.evidence)
            and bool(re.search(r'\d|₽|рубл|бесплат|стоим|тариф|комисс|лимит|процент|скид', claim.text.casefold())))


def validate_answer(answer: GroundedAnswer, chunks: list[dict]) -> None:
    """Проверяет принадлежность ссылок и наличие опор; не заменяет оценку смыслового обоснования."""
    mapping = {item['chunk_id']: item for item in chunks}
    if answer.supported != bool(answer.claims):
        raise ValueError('Статус ответа не соответствует утверждениям.')
    for claim in answer.claims:
        for evidence in claim.evidence:
            if evidence.chunk_id not in mapping:
                raise ValueError('Указан фрагмент вне найденного контекста.')
            if evidence.quote_id:
                selected = mapping[evidence.chunk_id].get('quote_options', {}).get(evidence.quote_id)
                if selected is None:
                    raise ValueError('Указан неизвестный идентификатор цитаты.')
                evidence.quote = selected
            quote = normalize_quote(evidence.quote)
            if len(quote) < 8 or quote not in normalize_quote(mapping[evidence.chunk_id]['text']):
                raise ValueError('Опорная цитата отсутствует в указанном фрагменте.')


def render_answer(answer: GroundedAnswer, chunks: list[dict]) -> tuple[str, list[dict]]:
    """Проверяет опоры и формирует ответ со ссылками на редакцию и страницу документа."""
    validate_answer(answer, chunks)
    if not answer.supported:
        if answer.clarification.strip():
            return answer.clarification.strip(), []
        return 'В загруженных документах на выбранную дату недостаточно информации для ответа.', []
    mapping = {item['chunk_id']: item for item in chunks}
    citations, lines = {}, []
    for claim in answer.claims:
        labels = []
        for evidence in claim.evidence:
            item = mapping[evidence.chunk_id]
            label = f"[{item['source_id']}, с. {item['page']}]"
            if label not in labels:
                labels.append(label)
            citations[evidence.chunk_id] = {
                'chunk_id': item['chunk_id'], 'source_id': item['source_id'],
                'path': item['source_path'], 'page': item['page'], 'quote': evidence.quote,
            }
        lines.append(claim.text + ' ' + ' '.join(labels))
    return '\n'.join(lines), list(citations.values())


def model_record(stage: str, response, elapsed: float) -> dict:
    return {'stage': stage, 'usage': response.usage_metadata or {},
            'latency_ms': elapsed * 1000, 'content': response.content,
            'tool_calls': response.tool_calls, 'response_metadata': response.response_metadata}


class RAGConsultant:
    """Организует выбор инструмента моделью и ответ по найденным документам с проверкой цитат."""
    def __init__(self, model, index: HybridIndex):
        from atm_agent.context.budget import ContextBudget

        self.model, self.index = model, index
        self.answer_tokens = int(os.getenv('RAG_MAX_OUTPUT_TOKENS', '2048'))
        if self.answer_tokens <= 0:
            raise ValueError('RAG_MAX_OUTPUT_TOKENS должен быть положительным.')
        self.answer_model = model.model_copy(update={'max_tokens': self.answer_tokens}) if hasattr(model, 'model_copy') else model
        self.context = ContextBudget.from_env(model) if hasattr(model, 'tokens_count') else None
        self.trace_folder = None
        self.planner_prompt = (PROMPTS / 'rag_planner_prompt.txt').read_text()
        self.answer_prompt = (PROMPTS / 'rag_answer_prompt.txt').read_text()
        self.review_prompt = (PROMPTS / 'rag_review_prompt.txt').read_text()

    def consult(self, question: str, history: list[dict] | None = None, as_of: str = '', client_token: str = '',
                summary: str = '', summary_citations: list[str] | None = None) -> dict:
        """Выполняет до двух поисков и одну попытку исправления цитат; явная дата имеет приоритет."""
        from atm_agent.context.budget import ContextBudgetError

        if client_token and not re.fullmatch(r'[a-f0-9]{64}', client_token):
            raise ValueError('Некорректный служебный идентификатор трассировки.')
        if as_of:
            parse_date(as_of)
        traces, context, calls, records, answer_errors, review_errors = [], {}, [], [], [], []
        aliases = {}

        def invoke(stage, runnable, messages, schemas, structured=False):
            """Вызывает стадию RAG с проверкой бюджета и сохранением фактического расхода токенов."""
            try:
                preflight = self.context.check(stage, messages, schemas, self.context.rag_tokens) if self.context else None
                if preflight and structured:
                    preflight['max_output_tokens'] = self.answer_tokens
                start = time.perf_counter()
                output = runnable.invoke(messages)
            except ContextBudgetError as exc:
                exc.calls = list(records)
                raise
            except Exception as exc:
                raise ContextBudgetError(f'rag_call_failed:{type(exc).__name__}', records) from None
            raw = output['raw'] if structured else output
            elapsed = time.perf_counter() - start
            record = model_record(stage, raw, elapsed)
            if preflight:
                record.update(self.context.record(stage, raw, preflight, elapsed))
            records.append(record)
            if record.get('within_budget') is False:
                raise ContextBudgetError('actual_rag_budget_exceeded', records)
            return output

        @tool(args_schema=SearchInput)
        def search_knowledge(query: str, as_of: str = '', product: str = '',
                             document_id: str = '', source_id: str = '', include_undated: bool = False) -> str:
            """Найти фрагменты документов банка гибридным поиском с учётом даты редакции."""
            options = {'include_undated': True} if overview_question(question) and not as_of else {}
            result = self.index.search(query, as_of or date.today().isoformat(), product, document_id, source_id, **options)
            traces.append(result['trace'])
            undated = set(result['trace'].get('undated_sources', []))
            selected = []
            for item in result['chunks']:
                options = overview_options(item['text']) if item['source_id'] in undated else quote_options(item['text'])
                if not options:
                    continue
                context[item['chunk_id']] = {**item, 'quote_options': options}
                alias = next((key for key, value in aliases.items() if value == item['chunk_id']), None)
                if alias is None:
                    alias = f'c{len(aliases)}'
                    aliases[alias] = item['chunk_id']
                selected.append({**{key: value for key, value in context[item['chunk_id']].items()
                                    if key not in ('index_text', 'text')}, 'chunk_id': alias})
            return json.dumps({'as_of': result['trace'].get('as_of'), 'query': query,
                               'undated_sources': result['trace'].get('undated_sources', []),
                               'overview_only': bool(undated), 'chunks': selected}, ensure_ascii=False)

        catalog = [{key: item.get(key) for key in ('document_id', 'source_id', 'product', 'valid_from', 'valid_to')}
                   for item in self.index.documents]
        payload = {'question': question, 'requested_as_of': as_of or None,
                   'today': date.today().isoformat(), 'history': (history or []) if self.context else (history or [])[-8:],
                   'summary': summary, 'summary_citations': summary_citations or [], 'catalog': catalog}
        messages = [SystemMessage(self.planner_prompt), HumanMessage(json.dumps(payload, ensure_ascii=False))]
        plan = invoke('tool_selection', self.model.bind_tools([search_knowledge], tool_choice='auto'),
                      messages, [search_knowledge])
        if len(plan.tool_calls) > 2:
            raise ContextBudgetError('rag_tool_call_budget_exceeded', records)
        results = []
        for call in plan.tool_calls:
            if call['name'] != search_knowledge.name:
                raise ContextBudgetError('unknown_rag_tool', records)
            try:
                arguments = SearchInput.model_validate(call['args'])
                if as_of:
                    arguments.as_of = as_of
                arguments.include_undated = overview_question(question) and not arguments.as_of
                content = search_knowledge.invoke(arguments.model_dump())
            except Exception as exc:
                raise ContextBudgetError(f'rag_search_failed:{type(exc).__name__}', records) from None
            calls.append({'name': call['name'], 'arguments': arguments.model_dump()})
            results.append(ToolMessage(content=content, tool_call_id=call['id'], name=call['name']))
        if not calls or not context:
            answer = GroundedAnswer(supported=False)
        else:
            grounded_messages = [SystemMessage(self.answer_prompt), messages[1], plan, *results]
            generator = self.answer_model.with_structured_output(GroundedAnswer, include_raw=True)
            for attempt in range(2):
                output = invoke('grounded_answer', generator, grounded_messages, [GroundedAnswer], structured=True)
                try:
                    if output['raw'].response_metadata.get('finish_reason') == 'length':
                        raise ValueError('Ответ оборван лимитом: сократи ответ до двух утверждений и коротких цитат.')
                    if output.get('parsing_error') or not isinstance(output.get('parsed'), GroundedAnswer):
                        raise ValueError('Некорректная структура ответа.')
                    answer = output['parsed']
                    for claim in answer.claims:
                        for evidence in claim.evidence:
                            evidence.chunk_id = aliases.get(evidence.chunk_id, evidence.chunk_id)
                    validate_answer(answer, list(context.values()))
                    if answer.supported:
                        undated_sources = {source for trace in traces for source in trace.get('undated_sources', [])}
                        filtered = []
                        for index, claim in enumerate(answer.claims):
                            if undated_conditions(claim, context, undated_sources):
                                review_errors.append({'index': index, 'reason': 'Условия из недатированного документа запрещены в ознакомительном режиме.'})
                            else:
                                filtered.append(claim)
                        answer.claims = filtered
                        if not answer.claims:
                            raise ValueError('Для недатированного источника оставь только общее описание продукта, без тарифов и чисел.')
                        review_messages = [SystemMessage(
                            self.review_prompt),
                            HumanMessage(json.dumps({
                                'question': question, 'answer': answer.model_dump(),
                                'sources': [{'chunk_id': item['chunk_id'], 'text': item['text']}
                                            for item in context.values()],
                                'undated_sources': [source for trace in traces
                                                    for source in trace.get('undated_sources', [])],
                            }, ensure_ascii=False))]
                        reviewed = invoke('answer_verification', self.answer_model.with_structured_output(
                            SupportReview, include_raw=True), review_messages, [SupportReview], structured=True)
                        review = reviewed.get('parsed')
                        if reviewed.get('parsing_error') or not isinstance(review, SupportReview):
                            raise ValueError('Не удалось проверить смысловую опору ответа.')
                        if any(item.index >= len(answer.claims) for item in review.rejected):
                            raise ValueError('Проверка вернула неизвестное утверждение.')
                        review_errors.extend(item.model_dump() for item in review.rejected)
                        rejected = {item.index for item in review.rejected}
                        answer.claims = [claim for index, claim in enumerate(answer.claims) if index not in rejected]
                        if not answer.claims:
                            raise ValueError('Проверка отклонила ответ: ' + '; '.join(item.reason for item in review.rejected))
                    break
                except ValueError as exc:
                    answer_errors.append(str(exc))
                    if attempt:
                        answer = GroundedAnswer(supported=False)
                        break
                    grounded_messages.append(HumanMessage(
                        f'Исправь ответ: {exc} Используй только точные chunk_id и короткие дословные цитаты '
                        'из результатов инструмента. Сохрани полезное содержание ответа. '
                        'Выбери цитаты заново, не переписывай длинные абзацы.'))
        response, citations = render_answer(answer, list(context.values()))
        if answer_errors and not answer.supported and not answer.clarification:
            response = ('Не удалось сформировать проверенный ответ по найденным документам. '
                        'Попробуйте уточнить вопрос — например, назовите интересующий сервис или условие.')
        undated = {source for trace in traces for source in trace.get('undated_sources', [])}
        if any(citation['source_id'] in undated for citation in citations):
            response += ' Дата действия этого документа не установлена; актуальность описанных условий не подтверждена.'
        result = {'response': response, 'supported': answer.supported, 'citations': citations,
                  'retrieval_trace': traces, 'rag_tool_calls': calls, 'rag_calls': records,
                  'context': list(context.values()), 'pipeline': getattr(self.index, 'manifest', {}),
                  'model': self.model.model, 'created_at': datetime.now(timezone.utc).isoformat(),
                  'question': question, 'as_of': as_of or date.today().isoformat(), 'answer_errors': answer_errors,
                  'review_rejections': review_errors,
                  'memory_client_token': client_token or None}
        folder = self.trace_folder or INDEX / 'runs'
        folder.mkdir(parents=True, exist_ok=True)
        owner = client_token + '_' if client_token else ''
        path = folder / f'{owner}{uuid4().hex}.json'
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        result['trace_file'] = str(path)
        return result


def main():
    """Запускает отдельную консультацию по документам и сохраняет её трассировку."""
    parser = argparse.ArgumentParser(description='Консультация по документам через инструмент гибридного поиска')
    parser.add_argument('--question', required=True)
    parser.add_argument('--as-of', default='')
    parser.add_argument('--env-file')
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    model, embeddings = create_model(args.env_file), create_embeddings(args.env_file)
    index = HybridIndex(embeddings)
    try:
        result = RAGConsultant(model, index).consult(args.question, as_of=args.as_of)
        print(result['response'])
        print('Трассировка:', result['trace_file'])
        if args.trace:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        index.close()


if __name__ == '__main__':
    main()
