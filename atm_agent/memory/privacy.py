import asyncio
from threading import RLock


class ManagedGraph:
    """Завершает удаление после записи последнего checkpoint, затем стирает все сессии клиента."""

    def __init__(self, graph, checkpointer, memory, sessions, contexts=()):
        self.graph, self.checkpointer, self.memory, self.sessions = graph, checkpointer, memory, sessions
        self.lock = RLock()
        self.contexts = [context for context in contexts if context is not None]

    def invoke(self, data, config, **kwargs):
        # В одном банкомате ходы последовательны, включая чтение и удаление трассировок.
        """Выполняет ход последовательно и завершает удаление после записи checkpoint, исключая восстановление истории."""
        with self.lock:
            state = self.graph.invoke(data, config, **kwargs)
            if not state.get('erase_memory_requested'):
                return state
            client_id = state['client_id']
            try:
                report = self.memory.forget(client_id)
            except Exception as exc:
                return {**state, 'erase_memory_requested': False,
                        'response': 'Не удалось удалить память. Попробуйте позже.',
                        'memory_error': f'erase_failed:{type(exc).__name__}'}
            for thread_id in self.sessions.pop(client_id, set()):
                self.checkpointer.delete_thread(thread_id)
            for context in self.contexts:
                context.cache.clear()
            artifact_error = None
            try:
                report['artifacts_removed'] = self.memory.store.cleanup_artifacts()
            except Exception as exc:
                artifact_error = f'artifact_cleanup_failed:{type(exc).__name__}'
            response = ('Предпочтения, эпизоды, поисковые векторы и история сессий удалены. '
                        f'Минимальные служебные квитанции и аудит сохраняются до окончания '
                        f'учебного срока хранения ({report["retention_days"]} дней с даты записи).')
            if artifact_error:
                response += ' Очистка файлов трассировок ещё ожидает повторной обработки.'
            return {'client_id': client_id, 'response': response, 'usage': state.get('turn_usage', state.get('usage', {})),
                    'memory_error': artifact_error, 'deletion_report': report}

    async def ainvoke(self, data, config, **kwargs):
        return await asyncio.to_thread(self.invoke, data, config, **kwargs)

    def stream(self, *args, **kwargs):
        raise NotImplementedError('Для диалога с удалением данных используйте invoke или ainvoke.')

    def __getattr__(self, name):
        if name in ('get_state', 'get_state_history', 'update_state', 'get_graph', 'store', 'checkpointer'):
            return getattr(self.graph, name)
        raise AttributeError(name)
