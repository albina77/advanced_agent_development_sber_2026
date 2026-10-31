import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from atm_agent.atm.mock import DemoATMSoftware
from atm_agent.cli.chat import session_config
from atm_agent.config.paths import DATA, REPOSITORY
from atm_agent.config.settings import create_embeddings, create_model
from atm_agent.dialogue.graph import build_graph
from atm_agent.memory.client import ClientMemory
from atm_agent.memory.store import SqliteStore
from atm_agent.understanding.extractor import GigaChatExtractor


def run(env_file=None):
    """Живая проверка критерия 6 с отдельным процессом индексации и временной базой."""
    if env_file:
        env_file = str(Path(env_file).expanduser().resolve())
    model, embeddings = create_model(env_file), create_embeddings(env_file)
    now = datetime.now(timezone.utc)
    current_time = now - timedelta(days=10)
    report = {'passed': False, 'turns': [], 'clock_is_simulated': True}
    with tempfile.TemporaryDirectory(prefix='atm-memory-lifecycle-') as folder:
        path = Path(folder) / 'store.sqlite3'
        store = SqliteStore(path, embeddings=embeddings, clock=lambda: current_time)
        memory = ClientMemory(store, clock=lambda: current_time)
        software = DemoATMSoftware({})
        app = build_graph(GigaChatExtractor(model), software, memory=memory)
        config = session_config('demo_lifecycle')

        def step(text):
            state = app.invoke({'text': text}, config)
            print('Вы:', text, flush=True)
            if state.get('response'):
                print('Агент:', state['response'], flush=True)
            if state.get('error') or state.get('memory_error'):
                raise AssertionError('Ошибка диалога или памяти.')
            report['turns'].append({key: state.get(key) for key in (
                'response', 'extracted', 'usage', 'memory_matches', 'memory_index_status', 'deletion_report',
            )} | {'text': text, 'commands': len(software.commands)})
            return state

        def worker():
            command = [sys.executable, '-m', 'atm_agent.memory.worker', '--memory-db', str(path)]
            if env_file:
                command += ['--env-file', env_file]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180, cwd=REPOSITORY)
            if completed.returncode:
                raise AssertionError('Отдельный обработчик памяти не завершился успешно.')
            return json.loads(completed.stdout)

        try:
            step('Сними 7000 рублей купюрами по 1000 рублей')
            step('Да')
            old_request = software.commands[-1].request_id
            current_time = now
            config = session_config('demo_lifecycle')
            step('Сними 7000 рублей купюрами по 1000 рублей')
            step('Да')
            new_request = software.commands[-1].request_id
            step('Запомни: обычно выдавай купюрами по 500 рублей')
            assert memory.choice('demo_lifecycle')['denomination'] == 500
            assert store.index_status('demo_lifecycle')['pending'] == 3
            report['worker'] = worker()
            assert report['worker']['indexed'] == 3
            state = step('Что ты помнишь о моих предпочтениях и прошлых снятиях?')
            assert state['extracted']['intent'] == 'memory_recall'
            scores = {item['key']: item for item in state['memory_matches']}
            assert old_request in scores and new_request in scores
            assert scores[new_request]['score'] > scores[old_request]['score']
            report['freshness_comparison'] = {'older': scores[old_request], 'newer': scores[new_request]}
            state = step('Удали всю память обо мне, включая предпочтения и историю')
            assert state['deletion_report']['deleted_records'] == 3
            assert state['deletion_report']['retained_receipts'] == 2
            assert app.get_state(config).values == {}
            assert memory.choice('demo_lifecycle') is None
            assert store.search(('clients', 'demo_lifecycle'), query='предпочтения') == []
            step('Что ты помнишь обо мне?')
            report['worker_after_erasure'] = worker()
            assert report['worker_after_erasure']['indexed'] == 0
            assert len(software.commands) == 2
            report['receipts_after_erasure'] = store.receipts('demo_lifecycle')
            report['audit_after_erasure'] = store.audit('demo_lifecycle')
            assert all(set(receipt) == {'request_id', 'action', 'status', 'completed_at', 'retain_until'}
                       for receipt in report['receipts_after_erasure'])
            report['passed'] = True
        finally:
            store.close()
            if hasattr(embeddings, 'close'):
                embeddings.close()
            output = DATA / 'memory/lifecycle_report.json'
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Критерий 6: живой сценарий пройден; аудит, свежесть, отдельный worker и удаление проверены.', flush=True)


if __name__ == '__main__':
    from atm_agent.config.diagnostics import connection_error

    parser = argparse.ArgumentParser(description='Живой сценарий управления памятью без изменения базы клиента')
    parser.add_argument('--env-file')
    args = parser.parse_args()
    try:
        run(args.env_file)
    except Exception as exc:
        print(str(exc) if isinstance(exc, AssertionError) else connection_error(exc), flush=True)
        raise SystemExit(1)
