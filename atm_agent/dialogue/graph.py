from dataclasses import asdict

from langchain_core.runnables import RunnableConfig

from atm_agent.atm.contracts import ATMCommand, ATMSoftware
from atm_agent.dialogue.state import DialogueState, Turn, handle_turn, receive_result
from atm_agent.understanding.input_gate import Relevance


def build_graph(extractor, software: ATMSoftware, consultant=None, memory=None):
    """Собирает граф; изменяемое состояние каждой сессии находится в checkpointer."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    checkpointer = InMemorySaver()
    sessions = {}
    context = getattr(extractor, 'context', None)

    def prepare_context(state: DialogueState) -> dict:
        """Готовит окно истории и сбрасывает учёт текущего хода; при ошибке запрещает дальнейшее извлечение."""
        from atm_agent.context.budget import ContextBudgetError

        reset = {'summary_calls': [], 'model_calls': [], 'rag_calls': [], 'usage': {},
                 'context_error': None, 'response': None, 'outgoing': None, 'result': None,
                 'error': None, 'erase_memory_requested': False, 'citations': [],
                 'retrieval_trace': [], 'rag_tool_calls': [], 'rag_trace_file': None}
        try:
            return {**reset, **context.prepare(state, extractor.messages, extractor.schema)}
        except ContextBudgetError as exc:
            return {**reset, 'context_error': exc.reason, 'error': 'context_preparation_failed',
                    'summary_calls': exc.calls, 'context_stats': {'action': 'rejected', 'reason': exc.reason,
                                                               'input_budget': context.input_tokens}}

    def identify(state: DialogueState, config: RunnableConfig) -> dict:
        client_id = config.get('configurable', {}).get('client_id')
        memory.namespace(client_id, 'profile')
        if state.get('client_id') and state['client_id'] != client_id:
            raise ValueError('Нельзя менять клиента внутри сессии. Создайте новый thread_id.')
        sessions.setdefault(client_id, set()).add(config['configurable']['thread_id'])
        return {'client_id': client_id, 'memory_error': None, 'memory_event': None}

    def load_memory(state: DialogueState) -> dict:
        try:
            choice = memory.choice(state['client_id'])
            updates = {'memory_choice': choice, 'memory_available': True, 'memory_offer_invalidated': False}
        except Exception as exc:
            choice = None
            updates = {'memory_choice': None, 'memory_available': False, 'memory_offer_invalidated': False,
                       'memory_error': f'read_failed:{type(exc).__name__}'}
        if state.get('cash_offer') and state['cash_offer'] != choice:
            updates.update(cash_offer=None, cash_offer_active=False, cash_offer_declined=True,
                           memory_offer_invalidated=True)
        return updates

    def extract(state: DialogueState) -> dict:
        """Преобразует реплику в структурированные поля; ошибку извлечения трактует как неопределённый ввод."""
        from atm_agent.understanding.extractor import ExtractionError

        try:
            turn, usage = extractor.extract(state['text'], state)
            extracted = {**asdict(turn), 'relevance': turn.relevance.value}
            return {'extracted': extracted, 'usage': usage, 'response': None,
                    'outgoing': None, 'result': None, 'error': None,
                    'citations': [], 'retrieval_trace': [], 'rag_tool_calls': [],
                    'rag_calls': [], 'rag_supported': False, 'rag_trace_file': None,
                    'erase_memory_requested': False, 'memory_matches': [], 'memory_index_status': {},
                    'model_calls': list(getattr(extractor, 'last_calls', []))}
        except ExtractionError as exc:
            return {'extracted': {**asdict(Turn(Relevance.UNCERTAIN)), 'relevance': 'uncertain'}, 'usage': exc.usage,
                    'response': None, 'outgoing': None, 'result': None,
                    'error': exc.reason, 'citations': [], 'retrieval_trace': [],
                    'rag_tool_calls': [], 'rag_calls': [], 'rag_supported': False, 'rag_trace_file': None,
                    'erase_memory_requested': False, 'memory_matches': [], 'memory_index_status': {},
                    'model_calls': list(getattr(extractor, 'last_calls', []))}

    def route(state: DialogueState) -> str:
        if state['extracted']['relevance'] != Relevance.ADDRESSED:
            return 'ignore'
        return 'handle'

    def ignore(state: DialogueState) -> dict:
        # Чужая речь не попадает в содержательную историю и следующий промпт.
        return {'text': '', 'extracted': {
            'relevance': state['extracted']['relevance'], 'intent': 'other',
        }}

    def handle(state: DialogueState) -> dict:
        """Применяет намерение к состоянию операции и обрабатывает обращения к пользовательской памяти."""
        turn = Turn(**state['extracted'])
        if turn.intent in ('forget_memory', 'memory_recall'):
            updates = {'outgoing': None, 'confirmation_active': False, 'cash_offer_active': False}
            if state.get('device_pending'):
                return {**updates, 'response': 'Сначала дождитесь результата текущей операции, затем повторите просьбу о памяти.'}
            if memory is None:
                return {**updates, 'response': 'Память сейчас недоступна.'}
            if turn.intent == 'forget_memory':
                return {**updates, 'erase_memory_requested': True, 'slots': {}, 'pending': None,
                        'cash_offer': None, 'history': [], 'response': 'Удаляю память.'}
            try:
                result = memory.recall(state['client_id'], state['text'])
                return {**updates, 'response': result['response'], 'memory_matches': result['matches'],
                        'memory_index_status': result['index_status']}
            except Exception as exc:
                return {**updates, 'response': 'Не удалось выполнить поиск по памяти. Можно продолжить операцию.',
                        'memory_error': f'search_failed:{type(exc).__name__}'}
        if turn.intent == 'preference':
            if turn.preference_action == 'keep':
                return {'outgoing': None, 'response': 'Предпочтение остаётся без изменений.',
                        'confirmation_active': False, 'cash_offer_active': False}
            updates = {'outgoing': None, 'result': None, 'confirmation_active': False,
                       'cash_offer': None, 'cash_offer_active': False}
            if state.get('device_pending'):
                return {**updates, 'response': 'Ожидаю результат текущей операции от банкомата.'}
            if memory is None or not state.get('memory_available'):
                return {**updates, 'response': 'Память сейчас недоступна. Предпочтение не изменено.'}
            try:
                if turn.preference_action == 'clear':
                    memory.clear_preference(state['client_id'])
                    response = 'Предпочтение по купюрам удалено. История успешных выдач сохранена.'
                else:
                    if turn.notes is not None:
                        return {**updates, 'response': 'Для предпочтения укажите один номинал или выбор банкомата.'}
                    memory.set_preference(state['client_id'], turn.denomination, turn.atm_choice)
                    choice = f'купюры по {turn.denomination} ₽' if turn.denomination else 'выбор купюр банкоматом'
                    response = f'Сохранено предпочтение на {memory.preference_days} дней: {choice}. При снятии уточню ваш выбор.'
                if state.get('slots'):
                    response += ' Можно продолжить текущее снятие.'
                return {**updates, 'response': response, 'memory_choice': memory.choice(state['client_id']),
                        'preference_hint_shown': True, 'cash_offer_declined': False}
            except ValueError as exc:
                from atm_agent.atm.contracts import OperationError

                response = str(exc) if isinstance(exc, OperationError) else 'Не удалось сохранить предпочтение.'
                return {**updates, 'response': response}
            except Exception as exc:
                return {**updates, 'response': 'Не удалось изменить предпочтение. Попробуйте позже.',
                        'memory_error': f'write_failed:{type(exc).__name__}'}
        return handle_turn(state, turn)

    def after_handle(state: DialogueState) -> str:
        if consultant is not None and state['extracted']['intent'] == 'product' and not state.get('device_pending'):
            return 'consult'
        return 'execute'

    def consult(state: DialogueState) -> dict:
        """Передаёт вопрос консультанту и регистрирует принадлежность трассировки для последующего удаления."""
        try:
            options = {'client_token': memory.store.client_token(state['client_id'])} if memory is not None else {}
            if context is not None:
                options['summary'] = state.get('summary', '')
                options['summary_citations'] = state.get('summary_citations', [])
            result = consultant.consult(state['text'], state.get('history', []), **options)
            if memory is not None and result.get('trace_file'):
                memory.store.register_artifact(state['client_id'], result['trace_file'])
            return {key: result[key] for key in (
                'response', 'citations', 'retrieval_trace', 'rag_tool_calls', 'rag_calls',
            )} | {'rag_supported': result['supported'], 'rag_trace_file': result.get('trace_file')}
        except Exception as exc:
            return {'response': 'Не удалось получить ответ по документам. Можно вернуться к операции или попробовать позже.',
                    'error': f'rag_failed:{type(exc).__name__}', 'rag_calls': getattr(exc, 'calls', [])}

    def execute(state: DialogueState) -> dict:
        """Отправляет подготовленную команду устройству и применяет результат либо неизвестный статус при сбое."""
        if not state.get('outgoing'):
            return {}
        command = ATMCommand(**state['outgoing'])
        try:
            result = software.send(command)
            updates = receive_result(state, result)
        except Exception:
            # Команда могла дойти. Блокируем новые операции до выяснения статуса.
            return {'device_pending': state['outgoing'], 'outgoing': None,
                    'error': 'device_status_unknown',
                    'response': 'Не удалось узнать результат. Ожидаю уточнения статуса от банкомата.'}
        if memory is not None and command.action == 'withdraw' and result.status == 'completed':
            updates['memory_event'] = {'command': state['outgoing'], 'result': asdict(result)}
        return updates

    def save_memory(state: DialogueState) -> dict:
        """Записывает подтверждённую выдачу; сбой памяти не меняет успешный результат операции."""
        from atm_agent.memory.client import PREFERENCE_HINT

        event = state.get('memory_event')
        if not event:
            return {}
        try:
            memory.record_withdrawal(state['client_id'], event['command'], event['result'])
            updates = {'memory_event': None}
            if not state.get('preference_hint_shown'):
                updates.update(response=state['response'] + ' ' + PREFERENCE_HINT, preference_hint_shown=True)
            return updates
        except Exception as exc:
            return {'memory_event': None, 'memory_error': f'write_failed:{type(exc).__name__}'}

    def remember(state: DialogueState) -> dict:
        """Добавляет содержательный ход в историю и при необходимости подсказку о явном предпочтении."""
        from atm_agent.memory.client import PREFERENCE_HINT

        response = state.get('response')
        if state.get('erase_memory_requested'):
            return {'history': [], 'text': ''}
        updates = {}
        if (memory is not None and response and not state.get('preference_hint_shown')
                and state.get('cash_offer')):
            response += ' ' + PREFERENCE_HINT
            updates.update(response=response, preference_hint_shown=True)
        history = list(state.get('history', []))
        history.append({'role': 'user', 'content': state['text']})
        if response:
            history.append({'role': 'assistant', 'content': response})
        return {**updates, 'history': history}

    def finish_turn(state: DialogueState) -> dict:
        """Суммирует фактический расход всех стадий без повторного учёта и сохраняет журнал бюджета."""
        from langchain_core.messages.ai import add_usage

        from atm_agent.context.budget import total_usage

        records = [*state.get('summary_calls', []), *state.get('model_calls', []), *state.get('rag_calls', [])]
        if not state.get('model_calls') and state.get('usage'):
            records.append({'stage': 'extract', 'usage': state['usage']})
        usage = total_usage(records)
        session = add_usage(state.get('session_usage') or None, usage or None) if usage else state.get('session_usage', {})
        turn = state.get('turn_no', 0) + 1
        log = {'turn': turn, 'context': state.get('context_stats', {'action': 'unmanaged'}),
               'calls': [{key: record[key] for key in ('stage', 'usage', 'preflight', 'within_budget', 'latency_ms')
                          if key in record} for record in records], 'actual_usage': usage}
        return {'turn_no': turn, 'turn_usage': usage, 'session_usage': session,
                'budget_log': state.get('budget_log', []) + [log]}

    builder = StateGraph(DialogueState)
    for name, node in (
        ('extract', extract), ('ignore', ignore), ('handle', handle),
        ('consult', consult), ('execute', execute), ('remember', remember), ('finish_turn', finish_turn),
    ):
        builder.add_node(name, node)
    if memory is not None:
        builder.add_node('identify', identify)
        builder.add_node('load_memory', load_memory)
        builder.add_node('save_memory', save_memory)
        builder.add_edge(START, 'identify')
        builder.add_edge('identify', 'prepare_context' if context is not None else 'extract')
        builder.add_edge('load_memory', 'handle')
        builder.add_edge('execute', 'save_memory')
        builder.add_edge('save_memory', 'remember')
    else:
        builder.add_edge(START, 'prepare_context' if context is not None else 'extract')
        builder.add_edge('execute', 'remember')
    builder.add_conditional_edges('extract', route, {'ignore': 'ignore', 'handle': 'load_memory' if memory is not None else 'handle'})
    if context is not None:
        builder.add_node('prepare_context', prepare_context)
        builder.add_conditional_edges('prepare_context',
            lambda state: 'finish_turn' if state.get('context_error') else 'extract',
            {'finish_turn': 'finish_turn', 'extract': 'extract'})
    builder.add_edge('ignore', 'finish_turn')
    builder.add_conditional_edges('handle', after_handle, {'consult': 'consult', 'execute': 'execute'})
    builder.add_edge('consult', 'remember')
    builder.add_edge('remember', 'finish_turn')
    builder.add_edge('finish_turn', END)
    graph = builder.compile(checkpointer=checkpointer, store=memory.store if memory is not None else None)
    if memory is not None:
        from atm_agent.memory.privacy import ManagedGraph

        return ManagedGraph(graph, checkpointer, memory, sessions,
                            contexts=(context, getattr(consultant, 'context', None)))
    return graph
