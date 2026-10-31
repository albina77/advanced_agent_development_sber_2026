import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from atm_agent.atm.contracts import ATMCommand, ATMResult
from atm_agent.atm.mock import DemoATMSoftware
from atm_agent.cli.chat import session_config
from atm_agent.config.paths import PROJECT, REPOSITORY
from atm_agent.config.settings import create_model
from atm_agent.context.budget import total_usage
from atm_agent.dialogue.graph import build_graph
from atm_agent.knowledge.bootstrap import ensure_ready
from atm_agent.memory.client import ClientMemory
from atm_agent.memory.store import SqliteStore
from atm_agent.retrieval.consultant import RAGConsultant
from atm_agent.retrieval.index import HybridIndex
from atm_agent.understanding.extractor import GigaChatExtractor

EXAMPLES = PROJECT / 'examples' / 'memory'


def expectations(case_id, session, turn):
    """Задаёт структурные ожидания для неизменённых вымышленных реплик набора."""
    pending = lambda amount, denomination: {'pending': {'amount': amount, 'denomination': denomination},
                                            'confirmation_active': True}
    offer = lambda denomination: {'cash_offer': {'denomination': denomination}}
    withdraw = lambda amount, denomination: {'commands': [{'action': 'withdraw', 'amount': amount,
                                                          'denomination': denomination}]}
    rules = {
        'M01': [[pending(6000, 1000), withdraw(6000, 1000)],
                [{**offer(1000), 'slots': {'amount': 8000}}, pending(8000, 1000), withdraw(8000, 1000)]],
        'M02': [[{'preference': 500}, {**pending(4000, 1000), 'preference': 500},
                 {**withdraw(4000, 1000), 'preference': 500}],
                [{**offer(500), 'slots': {'amount': 7000}}, pending(7000, 500), withdraw(7000, 500)]],
        'M03': [[{'preference': 100}, {'preference': 1000}],
                [offer(1000), pending(9000, 1000), withdraw(9000, 1000)]],
        'M04': [[{'preference': 'atm'}],
                [{'cash_offer': None, 'slots': {'amount': 5000}}, pending(5000, None), {'slots': {}, 'pending': None}],
                [{'cash_offer': {'atm_choice': True}, 'slots': {'amount': 2000}}]],
        'M05': [[{'preference': 1000}, pending(3000, 1000), withdraw(3000, 1000), {'erased': True}],
                [{'cash_offer': None, 'preference': None, 'slots': {'amount': 4000}, 'erased_records': True}]],
        'M06': [[{'preference': 500}, pending(2000, 1000), withdraw(2000, 1000)],
                [{**offer(500), 'slots': {'amount': 6000}, 'preference': 500}],
                [{'cash_offer': None, 'slots': {'amount': 8000}, 'preference': None}]],
        'M07': [[{'freshness': True}, {'background': True}]],
    }
    if case_id != 'M08':
        return {'commands': [], **rules[case_id][session - 1][turn - 1]}
    if session == 2:
        return {'commands': [], **[
            {**offer(1000), 'slots': {'amount': 10000}}, pending(10000, 1000), withdraw(10000, 1000),
        ][turn - 1]}
    result = {'commands': []}
    if 2 <= turn <= 33:
        result['slots'] = {'amount': 12000}
    if turn in (4, 10, 17, 26):
        result['commands'] = [{'action': 'get_balance'}]
    if turn in (7, 14, 21, 31):
        result['background'] = True
    if turn in (5, 13):
        result['source'] = 'sberprime-offer@2026-03-01' if turn == 5 else 'sberprime-offer@2026-09-30'
    if turn >= 11:
        result['preference'] = 1000
    if turn == 18:
        result['recall_preference'] = True
    if turn in (30, 33):
        result.update(pending(12000, 500))
    if turn == 33:
        result.update(response_contains='12000 ₽', amount_not_repeated=True)
    if turn == 32:
        result['confirmation_active'] = False
    if turn == 34:
        result.update(withdraw(12000, 500), summarized=True)
    return result


def check_turn(expected, state, commands, memory, app, config, before):
    """Проверяет действия, состояние и память; не оценивает литературное качество ответа."""
    failures = []

    def check(ok, message):
        if not ok:
            failures.append(message)

    check(not state.get('error') and not state.get('memory_error'), 'Ошибка агента или памяти')
    expected_commands = expected['commands']
    check(len(commands) == len(expected_commands), f'Ожидалось команд: {len(expected_commands)}, получено: {len(commands)}')
    for actual, wanted in zip(commands, expected_commands):
        check(all(actual.get(key) == value for key, value in wanted.items()), f'Параметры команды отличаются: {wanted}')
    for key in ('slots', 'pending', 'cash_offer', 'confirmation_active'):
        if key not in expected:
            continue
        actual, wanted = state.get(key), expected[key]
        if isinstance(wanted, dict) and wanted:
            check(isinstance(actual, dict) and all(actual.get(k) == v for k, v in wanted.items()), f'{key}: ожидалось {wanted}, получено {actual}')
        else:
            check(actual == wanted, f'{key}: ожидалось {wanted}, получено {actual}')
    client = config['configurable']['client_id']
    if 'preference' in expected:
        choice = memory.choice(client)
        actual = ('atm' if choice['atm_choice'] else choice['denomination']) if choice else None
        check(actual == expected['preference'], f'Выбор из памяти: ожидалось {expected["preference"]}, получено {actual}')
    if expected.get('background'):
        check(state.get('response') is None, 'Фоновая реплика вызвала ответ')
        check(not any(item['content'] == state.get('text') for item in state.get('history', [])), 'Фоновая реплика попала в историю')
        if state.get('summary_revision', 0) == before.get('summary_revision', 0):
            check(state.get('history', []) == before.get('history', []), 'Фон изменил историю без суммаризации')
        for key in ('slots', 'pending', 'confirmation_active'):
            check(state.get(key) == before.get(key), f'Фон изменил {key}')
    if expected.get('source'):
        check(state.get('rag_supported') and any(item['source_id'] == expected['source'] for item in state.get('citations', [])), 'Нет подтверждённого ответа по нужной редакции')
    if expected.get('recall_preference'):
        check(any(item['kind'] == 'cash_preference' for item in state.get('memory_matches', [])), 'Предпочтение не найдено смысловым поиском')
    if expected.get('freshness'):
        matches = {item['key']: item for item in state.get('memory_matches', [])}
        check('older' in matches and 'newer' in matches, 'Найдены не оба подготовленных эпизода')
        if 'older' in matches and 'newer' in matches:
            check(abs(matches['older']['similarity'] - matches['newer']['similarity']) < 1e-6, 'Близость одинаковых эпизодов различается')
            check(matches['newer']['score'] > matches['older']['score'], 'Новый эпизод не получил преимущество по свежести')
    if expected.get('erased'):
        report = state.get('deletion_report', {})
        check(report.get('deleted_records') == 2, 'Удалены не обе пользовательские записи')
        check(report.get('retained_receipts') == 1, 'Не сохранена минимальная квитанция')
        check(app.get_state(config).values == {}, 'Checkpoint не очищен')
    if expected.get('erased_records') or expected.get('erased'):
        check(memory.store.search(('clients', client)) == [], 'Пользовательские записи не удалены или восстановились')
        check(memory.store.index_status(client) == {'ready': 0, 'pending': 0, 'failed': 0}, 'Остались записи семантического индекса')
    if expected.get('summarized'):
        check(state.get('summary_revision', 0) > 0, 'Длинная история не суммаризирована')
    if expected.get('response_contains'):
        check(expected['response_contains'] in (state.get('response') or ''), 'Агент не назвал сохранённую сумму')
    if expected.get('amount_not_repeated'):
        check(state.get('extracted', {}).get('amount') is None, 'Сумма должна браться из состояния, а не из текущей реплики')
    return failures


def run_worker(path, env_file):
    """Индексирует память отдельным процессом вне хода диалога."""
    command = [sys.executable, '-m', 'atm_agent.memory.worker', '--memory-db', str(path)]
    if env_file:
        command += ['--env-file', env_file]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=REPOSITORY, timeout=180)
    if completed.returncode:
        raise RuntimeError('Отдельный worker завершился с ошибкой')
    return json.loads(completed.stdout)


def save_report(report, output):
    """Сохраняет машинный отчёт и читаемый лог, включая неуспешные шаги."""
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_log.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Прогон вымышленных сценариев памяти', '',
             f'Начало: {report["started_at"]}. Модель: {report["model"]}.', '',
             'Использованы живые GigaChat и Cloud.ru, учебный софт и временные базы. '
             'Между сессиями Store закрывался и открывался заново. '
             'Проверки структурные; ожидаемый текст приведён для ручной оценки содержания ответа.', '']
    lines += ['| Сценарий | Статус | Пройдено реплик |', '|---|---|---|']
    for case in report['cases']:
        passed = sum(not row['failures'] for row in case['turns'])
        lines.append(f'| {case["id"]} | {case["status"]} | {passed}/{len(case["turns"])} |')
    lines.append('')
    for case in report['cases']:
        lines += [f'## {case["id"]}. {case["title"]}', '', f'Статус: {case["status"]}.', '',
                  'Подготовка: ' + case['setup'], '']
        for row in case['turns']:
            lines += [f'### Сессия {row["session"]}, реплика {row["turn"]}', '',
                      f'Клиент: {row["text"]}', '', f'Агент: {row.get("response") or "[молчание]"}', '',
                      f'Ожидание: {row["expected"]}', '', f'Проверка: {"FAIL" if row["failures"] else "PASS"}.', '']
            if row['commands']:
                lines += ['Команды: `' + json.dumps(row['commands'], ensure_ascii=False) + '`', '']
            for failure in row['failures']:
                lines += [f'- {failure}', '']
        if case.get('fatal_error'):
            lines += [f'Ошибка выполнения: {case["fatal_error"]}', '']
    lines += ['## Итог', '', json.dumps(report.get('summary', {}), ensure_ascii=False), '']
    (output / 'run_log.md').write_text('\n'.join(lines))


def run(env_file=None, case_ids=None, output=EXAMPLES):
    """Прогоняет исходные 70 реплик без исправления формулировок и не останавливается на провале ожидания."""
    if env_file:
        env_file = str(Path(env_file).expanduser().resolve())
    dataset_bytes = (EXAMPLES / 'scenarios.json').read_bytes()
    dataset = json.loads(dataset_bytes)
    model = create_model(env_file)
    embeddings, reranker = ensure_ready(env_file)
    index = HybridIndex(embeddings, reranker=reranker)
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'model': model.model,
              'synthetic': True, 'dataset_sha256': hashlib.sha256(dataset_bytes).hexdigest(),
              'status': 'running', 'cases': []}
    try:
        for case in dataset['cases']:
            if case_ids and case['id'] not in case_ids:
                continue
            result = {'id': case['id'], 'title': case['title'], 'setup': case['setup'],
                      'status': 'running', 'turns': [], 'workers': []}
            report['cases'].append(result)
            with tempfile.TemporaryDirectory(prefix='atm-synthetic-') as folder:
                path = Path(folder) / 'store.sqlite3'
                now = datetime.now(timezone.utc)
                current = now
                software_by_client = {}
                consultant = RAGConsultant(model, index)
                consultant.trace_folder = Path(folder) / 'rag_traces'
                try:
                    if case['id'] == 'M07':
                        with_store = SqliteStore(path, embeddings=embeddings, clock=lambda: current)
                        try:
                            memory = ClientMemory(with_store, clock=lambda: current)
                            for request, age in [('older', 12), ('newer', 0)]:
                                current = now - timedelta(days=age)
                                memory.record_withdrawal('synthetic_client_h', asdict(ATMCommand(request, 'withdraw', 6000, denomination=1000)),
                                                         asdict(ATMResult(request, 'completed', amount=6000)))
                        finally:
                            with_store.close()
                        current = now
                        result['workers'].append(run_worker(path, env_file))
                    for session_number, session in enumerate(case['sessions'], 1):
                        current = now + timedelta(days=session.get('clock_offset_days', 0))
                        store = SqliteStore(path, embeddings=embeddings, clock=lambda: current, artifact_root=consultant.trace_folder)
                        try:
                            memory = ClientMemory(store, clock=lambda: current)
                            client = session['client_id']
                            software = software_by_client.setdefault(client, DemoATMSoftware())
                            app = build_graph(GigaChatExtractor(model), software, consultant, memory)
                            config = session_config(client)
                            for number, turn in enumerate(session['turns'], 1):
                                if (case['id'], session_number, number) in [('M08', 1, 18), ('M05', 2, 1)]:
                                    result['workers'].append(run_worker(path, env_file))
                                before = app.get_state(config).values
                                command_count = len(software.commands)
                                state = app.invoke({'text': turn['text']}, config)
                                commands = [asdict(item) for item in software.commands[command_count:]]
                                spec = expectations(case['id'], session_number, number)
                                failures = check_turn(spec, state, commands, memory, app, config, before)
                                row = {'session': session_number, 'turn': number, 'client_id': client,
                                       'text': turn['text'], 'expected': turn['expected'], 'assertions': spec,
                                       'clock': current.isoformat(),
                                       'commands': commands, 'failures': failures,
                                       **{key: state.get(key) for key in ('response', 'extracted', 'slots', 'pending', 'cash_offer',
                                           'confirmation_active', 'memory_matches', 'deletion_report', 'summary_revision',
                                           'summary', 'citations', 'rag_supported', 'turn_usage', 'error', 'memory_error')}}
                                row['turn_usage'] = state.get('turn_usage') or state.get('usage', {})
                                result['turns'].append(row)
                                print(f'{case["id"]} {session_number}.{number}: {"FAIL" if failures else "PASS"} — {turn["text"]}', flush=True)
                                for failure in failures:
                                    print('  ' + failure, flush=True)
                                save_report(report, output)
                        finally:
                            store.close()
                    result['status'] = 'failed' if any(row['failures'] for row in result['turns']) else 'passed'
                except Exception as exc:
                    result['status'] = 'error'
                    result['fatal_error'] = type(exc).__name__
                save_report(report, output)
    finally:
        index.close()
        if hasattr(embeddings, 'close'):
            embeddings.close()
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['status'] = 'finished'
        rows = [row for case in report['cases'] for row in case['turns']]
        report['summary'] = {'cases': len(report['cases']), 'passed_cases': sum(case['status'] == 'passed' for case in report['cases']),
                             'turns': len(rows), 'passed_turns': sum(not row['failures'] for row in rows),
                             'usage': total_usage([{'usage': row.get('turn_usage')} for row in rows])}
        save_report(report, output)
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)
    return all(case['status'] == 'passed' for case in report['cases'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Живой прогон вымышленных сценариев памяти')
    parser.add_argument('--env-file')
    parser.add_argument('--case', action='append', choices=[f'M{i:02}' for i in range(1, 9)])
    parser.add_argument('--output', type=Path, default=EXAMPLES)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.env_file, args.case, args.output) else 1)
