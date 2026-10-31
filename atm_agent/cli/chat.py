import argparse
import json
import sys
from dataclasses import asdict
from uuid import uuid4

from atm_agent.atm.contracts import OperationError
from atm_agent.atm.mock import DemoATMSoftware
from atm_agent.config.diagnostics import connection_error
from atm_agent.config.paths import DATA
from atm_agent.dialogue.state import receive_result


def session_config(client_id: str) -> dict:
    return {'configurable': {'thread_id': uuid4().hex, 'client_id': client_id}}


def print_trace(state: dict, commands: list) -> None:
    """Выводит состояние операции, команды, ошибки и расход токенов для диагностики хода."""
    print(json.dumps({
        'extracted': state.get('extracted'),
        'slots': state.get('slots', {}),
        'awaiting_confirmation': state.get('confirmation_active', False),
        'device_pending': state.get('device_pending'),
        'commands': [asdict(command) for command in commands],
        'result': state.get('result'),
        'usage': state.get('usage'),
        'turn_usage': state.get('turn_usage'),
        'session_usage': state.get('session_usage'),
        'summary': state.get('summary', ''),
        'summary_citations': state.get('summary_citations', []),
        'summary_revision': state.get('summary_revision', 0),
        'context_stats': state.get('context_stats', {}),
        'context_error': state.get('context_error'),
        'budget': (state.get('budget_log') or [None])[-1]
        if state.get('extracted', {}).get('source') != 'device_event' else None,
        'error': state.get('error'),
        'client_id': state.get('client_id'),
        'memory_choice': state.get('memory_choice'),
        'cash_offer': state.get('cash_offer'),
        'cash_offer_active': state.get('cash_offer_active', False),
        'memory_error': state.get('memory_error'),
        'memory_matches': state.get('memory_matches', []),
        'memory_index_status': state.get('memory_index_status', {}),
        'deletion_report': state.get('deletion_report'),
        'citations': state.get('citations', []),
        'retrieval_trace': state.get('retrieval_trace', []),
        'rag_tool_calls': state.get('rag_tool_calls', []),
        'rag_usage': [{'stage': call['stage'], 'usage': call['usage']} for call in state.get('rag_calls', [])],
        'rag_trace_file': state.get('rag_trace_file'),
    }, ensure_ascii=False, indent=2))


def run_scenario(app, config: dict, software: DemoATMSoftware, trace: bool) -> int:
    """Проверяет короткий проход на живом GigaChat и заглушке устройства."""
    utterances = [
        'Хочу снять', 'Семь тысяч',
        'Уважаемые пассажиры! Объявляется посадка на рейс до Москвы.',
        'Тысячными', 'Да',
    ]
    for index, text in enumerate(utterances):
        before = len(software.commands)
        state = app.invoke({'text': text}, config)
        print('Вы:', text)
        if state.get('response'):
            print('Агент:', state['response'])
        if trace:
            print_trace(state, software.commands[before:])
        if state.get('error') or (index < 4 and software.commands):
            print('Сценарий не пройден: ошибка извлечения или преждевременная команда.')
            return 1
        if index == 2 and state.get('response') is not None:
            print('Сценарий не пройден: агент ответил на объявление.')
            return 1
    if len(software.commands) != 1:
        print('Сценарий не пройден: ожидалась одна команда на выдачу.')
        if not trace:
            print_trace(state, software.commands)
        return 1
    command = software.commands[0]
    if (command.action, command.amount, command.denomination) != ('withdraw', 7000, 1000):
        print('Сценарий не пройден: неверные параметры команды.')
        return 1
    print('Сценарий пройден: одна команда, 7000 ₽ тысячными, после подтверждения.')
    return 0


def main() -> int:
    """Запускает выбранную проверку или текстовый диалог с учебным софтом банкомата."""
    parser = argparse.ArgumentParser(description='Текстовый агент банкомата, GigaChat + LangGraph')
    parser.add_argument('--env-file')
    parser.add_argument('--trace', action='store_true', help='Показывать параметры и команды')
    parser.add_argument('--check', action='store_true', help='Проверить один запрос к GigaChat')
    parser.add_argument('--scenario', action='store_true', help='Прогнать пример на живой модели')
    parser.add_argument('--question', help='Выполнить одну консультацию RAG и выйти')
    parser.add_argument('--as-of', default='', help='Дата для одиночной консультации YYYY-MM-DD')
    parser.add_argument('--client-id', default='demo_albina', help='Имитация идентификатора от считывателя карты')
    parser.add_argument('--memory-db', help='Путь к SQLite Store (по умолчанию data/memory/store.sqlite3)')
    args = parser.parse_args()
    try:
        from atm_agent.config.settings import create_model

        model = create_model(args.env_file)
        if args.check:
            print(f'Модель: {model.model}; scope: {model.scope}', flush=True)
            response = model.invoke('Ответь одним словом: подключено')
            if not response.content:
                raise ValueError('Модель вернула пустой ответ.')
            print('Ответ GigaChat:', response.content)
            print('Использование токенов:', json.dumps(response.usage_metadata, ensure_ascii=False))
            return 0
        from atm_agent.dialogue.graph import build_graph
        from atm_agent.understanding.extractor import GigaChatExtractor

        software = DemoATMSoftware({})
        from atm_agent.config.settings import create_embeddings
        from atm_agent.knowledge.corpus import INDEX
        from atm_agent.retrieval.consultant import RAGConsultant
        from atm_agent.retrieval.index import HybridIndex

        consultant = None
        embeddings = None
        if not args.scenario:
            from atm_agent.knowledge.bootstrap import ensure_ready

            embeddings, reranker = ensure_ready(args.env_file)
            consultant = RAGConsultant(model, HybridIndex(embeddings, reranker=reranker))
        elif (INDEX / 'index.json').exists():
            consultant = RAGConsultant(model, HybridIndex(create_embeddings(args.env_file)))
        if args.question:
            if consultant is None:
                print('Сначала постройте индекс: python -m atm_agent.knowledge.build_index', file=sys.stderr)
                return 1
            try:
                result = consultant.consult(args.question, as_of=args.as_of)
                print(result['response'])
                print('Трассировка:', result['trace_file'])
                if args.trace:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0
            finally:
                consultant.index.close()
        import os
        from pathlib import Path

        from atm_agent.memory.client import ClientMemory
        from atm_agent.memory.store import SqliteStore

        store = SqliteStore(Path(args.memory_db) if args.memory_db else DATA / 'memory/store.sqlite3',
                            embeddings=embeddings or create_embeddings(args.env_file),
                            retention_days=int(os.getenv('MEMORY_SERVICE_RETENTION_DAYS', '365')),
                            min_similarity=float(os.getenv('MEMORY_MIN_SIMILARITY', '0.10')))
        memory = ClientMemory(store, preference_days=int(os.getenv('MEMORY_PREFERENCE_DAYS', '180')),
                              episode_days=int(os.getenv('MEMORY_EPISODE_DAYS', '30')))
        memory.namespace(args.client_id, 'profile')
        app = build_graph(GigaChatExtractor(model), software, consultant, memory)
    except ImportError:
        print('Не установлены зависимости. Запустите ./run.sh или pip install -r requirements.txt.',
              file=sys.stderr)
        return 1
    except Exception as exc:
        print(connection_error(exc), file=sys.stderr)
        return 1

    config = session_config(args.client_id)
    print(f'Считыватель карты — имитация: клиент {args.client_id}.')
    print('Софт банкомата — симулятор: начальный баланс 50000 ₽; снятия и внесения меняют остаток.')
    if args.scenario:
        return run_scenario(app, config, software, args.trace)
    print('Агент: Здравствуйте! Помогу снять или внести наличные, показать баланс '
          'и проконсультировать по СберПрайм, СберСпасибо и СберКарте.')
    print('/exit — выход, /new — новая сессия, /accepted 12500 — событие купюроприёмника.')
    while True:
        try:
            text = input('Вы: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text == '/exit':
            return 0
        if text == '/new':
            current = app.get_state(config).values
            if current.get('device_pending'):
                print('Сначала дождитесь результата операции банкомата.')
                continue
            config = session_config(args.client_id)
            print('Новая сессия того же клиента. Предпочтение и прошлый выбор доступны из памяти.')
            continue
        before = len(software.commands)
        if text.startswith('/accepted'):
            try:
                state = app.get_state(config).values
                pending = state.get('device_pending')
                if not pending or pending['action'] != 'start_deposit':
                    raise OperationError('Нет ожидающего внесения.')
                parts = text.split()
                if len(parts) != 2:
                    raise OperationError('Формат: /accepted 12500')
                result = software.complete_deposit(pending['request_id'], int(parts[1]))
                updates = receive_result(state, result)
                updates.update(usage={}, turn_usage={}, model_calls=[], summary_calls=[], rag_calls=[],
                               context_stats={'action': 'device_event'}, extracted={'source': 'device_event'})
                updates['history'] = state.get('history', []) + [
                    {'role': 'assistant', 'content': updates['response']},
                ]
                app.update_state(config, updates, as_node='remember')
                state = app.get_state(config).values
            except (ValueError, KeyError) as exc:
                print('Ошибка события устройства:', str(exc))
                continue
        else:
            try:
                state = app.invoke({'text': text}, config)
            except Exception as exc:
                print(f'Ошибка диалогового контура ({type(exc).__name__}).', file=sys.stderr)
                continue
        if state.get('response'):
            print('Агент:', state['response'])
        if state.get('error') == 'extraction_failed':
            print('Диагностика: параметры не получены от модели; команда не отправлена.', file=sys.stderr)
        elif state.get('context_error') or state.get('error') in ('context_budget_exceeded', 'actual_context_budget_exceeded'):
            print('Диагностика: контекст не подготовлен в пределах бюджета; команда не отправлена.', file=sys.stderr)
        if args.trace:
            print_trace(state, software.commands[before:])


if __name__ == '__main__':
    raise SystemExit(main())
