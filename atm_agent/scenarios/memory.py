import argparse
import json
import tempfile
from pathlib import Path

from atm_agent.atm.mock import DemoATMSoftware
from atm_agent.cli.chat import session_config
from atm_agent.config.paths import DATA
from atm_agent.config.settings import create_model
from atm_agent.dialogue.graph import build_graph
from atm_agent.memory.client import ClientMemory
from atm_agent.memory.store import SqliteStore
from atm_agent.understanding.extractor import GigaChatExtractor


def run(env_file=None):
    """Проверяет живую модель, повторное открытие Store и два подтверждения."""
    model = create_model(env_file)
    extractor, software = GigaChatExtractor(model), DemoATMSoftware({})
    turns = []

    def step(app, config, text, command_count):
        """Выполняет реплику живого сценария и сохраняет состояние для проверки памяти между сессиями."""
        state = app.invoke({'text': text}, config)
        print('Вы:', text, flush=True)
        if state.get('response'):
            print('Агент:', state['response'], flush=True)
        turns.append({key: state.get(key) for key in (
            'client_id', 'extracted', 'slots', 'pending', 'cash_offer', 'confirmation_active',
            'response', 'usage', 'memory_error', 'result',
        )} | {'text': text, 'command_count': len(software.commands)})
        if state.get('error') or state.get('memory_error'):
            raise AssertionError('Ошибка извлечения или памяти.')
        if len(software.commands) != command_count:
            raise AssertionError('Неверное число команд банкомату.')
        return state

    try:
        with tempfile.TemporaryDirectory(prefix='atm-memory-') as folder:
            path = Path(folder) / 'store.sqlite3'
            store = SqliteStore(path)
            try:
                app = build_graph(extractor, software, memory=ClientMemory(store))
                config = session_config('demo_memory')
                step(app, config, 'Сними 7000 рублей купюрами по 1000 рублей', 0)
                step(app, config, 'Да', 1)
            finally:
                store.close()

            store = SqliteStore(path)
            try:
                memory = ClientMemory(store)
                app = build_graph(extractor, software, memory=memory)
                config = session_config('demo_memory')
                state = step(app, config, 'Сними 9000 рублей', 1)
                assert state['cash_offer']['kind'] == 'withdrawal_episode'
                assert state['slots']['amount'] == 9000
                state = step(app, config, 'Да', 1)
                assert state['confirmation_active']
                step(app, config, 'Да', 2)
                assert software.commands[-1].amount == 9000
                step(app, config, 'Запомни: обычно выдавай купюрами по 500 рублей', 2)
                assert memory.choice('demo_memory')['denomination'] == 500
            finally:
                store.close()

            store = SqliteStore(path)
            try:
                memory = ClientMemory(store)
                app = build_graph(extractor, software, memory=memory)
                config = session_config('demo_memory')
                state = step(app, config, 'Сними 8000 рублей', 2)
                assert state['cash_offer']['kind'] == 'cash_preference'
                assert state['cash_offer']['denomination'] == 500
                state = step(app, config, 'Для этого снятия хочу купюрами по 1000 рублей', 2)
                assert state['pending']['denomination'] == 1000
                step(app, config, 'Да', 3)
                assert memory.choice('demo_memory')['denomination'] == 500
                state = step(app, session_config('other_client'), 'Сними 8000 рублей', 3)
                assert state.get('cash_offer') is None
            finally:
                store.close()
    except Exception:
        write_report(turns, False)
        raise
    write_report(turns, True)
    print('Сценарий памяти пройден: 3 подтверждённые выдачи, профиль и эпизоды разделены.', flush=True)


def write_report(turns, passed):
    folder = DATA / 'memory'
    folder.mkdir(parents=True, exist_ok=True)
    report = {'passed': passed, 'turns': turns,
              'checks': ['store_reopen', 'separate_confirmations', 'explicit_preference_priority',
                         'one_off_choice', 'client_isolation']}
    (folder / 'scenario_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    from atm_agent.config.diagnostics import connection_error

    parser = argparse.ArgumentParser(description='Живая проверка памяти на временной учебной базе')
    parser.add_argument('--env-file')
    args = parser.parse_args()
    try:
        run(args.env_file)
    except Exception as exc:
        print(str(exc) if isinstance(exc, AssertionError) else connection_error(exc), flush=True)
        raise SystemExit(1)
