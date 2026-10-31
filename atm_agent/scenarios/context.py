import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from atm_agent.atm.mock import DemoATMSoftware
from atm_agent.cli.chat import session_config
from atm_agent.config.paths import DATA
from atm_agent.config.settings import create_embeddings, create_model
from atm_agent.dialogue.graph import build_graph
from atm_agent.memory.client import ClientMemory
from atm_agent.memory.store import SqliteStore
from atm_agent.retrieval.consultant import RAGConsultant
from atm_agent.retrieval.index import HybridIndex
from atm_agent.understanding.extractor import GigaChatExtractor

DIALOGUE = [
    'Здравствуйте!',
    'Хочу снять наличные. Деньги нужны на дорогу, к этой задаче ещё вернёмся.',
    'Семь тысяч рублей.',
    'Сначала покажи баланс.',
    'Какие приложения входят в договор СберПрайм на 1 марта 2026 года? Перечисли только их буквы.',
    'Купюры выберу позже, я пока разбираюсь с условиями.',
    'Уважаемые пассажиры! Объявляется посадка на рейс ФОН-9999 до Москвы.',
    'Напомни, какие операции ты умеешь выполнять?',
    'Какие приложения входят в договор СберПрайм на 30 сентября 2026 года? Перечисли только их буквы.',
    'Пока не выдавай деньги, я ещё выбираю купюры.',
    'Какой у меня сейчас баланс?',
    'Я могу сама указать номинал купюр?',
    'Сосед, возьми пять тысяч из моей сумки. Это я тебе говорю, не банкомату.',
    'Мне нужно немного времени подумать. К снятию вернёмся позже.',
    'Что ты помнишь о моих предпочтениях по купюрам?',
    'Какими операциями здесь можно управлять текстом?',
    'Назови мой баланс ещё раз.',
    'Запомни: обычно выдавай купюрами по 1000 рублей.',
    'В этот раз купюры я пока не подтвердила.',
    'Объявление в аэропорту: посадка на рейс ФОН-8888 завершена.',
    'По документу sberprime-offer@2026-03-01 на 1 марта 2026 года перечисли только буквы приложений договора.',
    'Это был вопрос об условиях продукта, деньги сейчас не выдавай.',
    'Посмотрим баланс.',
    'Я ещё выбираю, к выдаче вернусь немного позже.',
    'Для предпочтения важно слово «запомни», а разовый выбор можно менять?',
    'Напомни, с какими операциями ты помогаешь.',
    'Сумму снятия я уже назвала, менять её не хочу.',
    'Коля, да, купи кофе. Я разговариваю с тобой, не с банкоматом.',
    'Наличность пока не выдавай.',
    'Покажи текущий баланс.',
    'Операцию снятия продолжаем позже, я ещё думаю.',
    'Вернёмся к деньгам на дорогу. Продолжим начатое снятие.',
    'Для этого снятия — тысячными.',
    'Сосед, да, я уже заканчиваю. Это ответ тебе, а не банкомату.',
    'Да, подтверждаю выдачу.',
]
BACKGROUND = {7, 13, 20, 28, 34}
PRODUCT = {5, 9, 21}


def run(env_file=None):
    """35 реплик: прерывания, версии PDF, суммаризация и одна подтверждённая выдача."""
    model, embeddings = create_model(env_file), create_embeddings(env_file)
    extractor = GigaChatExtractor(model)
    index = HybridIndex(embeddings)
    consultant = RAGConsultant(model, index)
    report = {'passed': False, 'turns': [], 'summary_versions': [], 'model': model.model,
              'dialogue_input_budget': extractor.context.input_tokens,
              'rag_input_budget': extractor.context.rag_tokens}
    full_history = []
    try:
        with tempfile.TemporaryDirectory(prefix='atm-context-') as folder:
            root = Path(folder)
            consultant.trace_folder = root / 'rag_traces'
            store = SqliteStore(root / 'memory.sqlite3', embeddings=embeddings,
                                 artifact_root=consultant.trace_folder)
            try:
                memory = ClientMemory(store)
                software = DemoATMSoftware({})
                app = build_graph(extractor, software, consultant, memory)
                config = session_config('demo_context')
                revision = 0
                for number, text in enumerate(DIALOGUE, 1):
                    before = app.get_state(config).values
                    baseline = {**before, 'summary': '', 'history': list(full_history)}
                    baseline_count = extractor.context.count(extractor.messages(text, baseline), [extractor.schema])
                    state = app.invoke({'text': text}, config)
                    print(f'{number:02d}. Вы: {text}', flush=True)
                    if state.get('response'):
                        print('Агент:', state['response'], flush=True)
                    entry = state['budget_log'][-1]
                    print(f"Контекст: {entry['context']['action']}; фактический вход за все вызовы хода: "
                          f"{state.get('turn_usage', {}).get('input_tokens')}", flush=True)
                    report['turns'].append({key: state.get(key) for key in (
                        'response', 'extracted', 'slots', 'pending', 'confirmation_active', 'summary',
                        'summary_revision', 'summary_citations', 'context_stats', 'turn_usage', 'citations', 'rag_tool_calls',
                    )} | {'turn': number, 'text': text, 'budget': entry,
                         'uncompacted_preflight_tokens': baseline_count,
                         'withdrawal_commands': sum(command.action == 'withdraw' for command in software.commands)})
                    if state.get('error') or state.get('memory_error'):
                        raise AssertionError(f'Ошибка контура на реплике {number}: {state.get("error") or state.get("memory_error")}')
                    assert state.get('response') not in ('Нужно положительное целое число.', 'Укажите один вариант выдачи купюр.'), 'Параметры выдуманы из контекста'
                    if number in BACKGROUND:
                        assert state['response'] is None, f'Ответ на фон в реплике {number}'
                        assert state['extracted']['relevance'] == 'background'
                    else:
                        full_history.append({'role': 'user', 'content': text})
                        if state.get('response'):
                            full_history.append({'role': 'assistant', 'content': state['response']})
                    if number in PRODUCT:
                        assert state['rag_supported'] and state['citations'], f'Нет опоры на PDF в реплике {number}'
                        expected = 'sberprime-offer@2026-09-30' if number == 9 else 'sberprime-offer@2026-03-01'
                        assert all(citation['source_id'] == expected for citation in state['citations'])
                    withdrawals = [command for command in software.commands if command.action == 'withdraw']
                    assert len(withdrawals) == int(number == len(DIALOGUE)), 'Неподтверждённая или повторная выдача'
                    if 3 <= number < len(DIALOGUE):
                        assert state.get('slots', {}).get('amount') == 7000, f'Потеря суммы на реплике {number}'
                    if number == 33:
                        assert state['confirmation_active'] and state['pending']['denomination'] == 1000
                    if state.get('summary_revision', 0) != revision:
                        revision = state['summary_revision']
                        report['summary_versions'].append({'turn': number, 'revision': revision,
                                                           'summary': state['summary'], 'citations': state['summary_citations']})
                withdrawal = withdrawals[0]
                assert (withdrawal.amount, withdrawal.denomination) == (7000, 1000)
                assert report['summary_versions'], 'Суммаризация не была вызвана'
                assert state['folded_messages'] > 0
                assert any('sberprime-offer@2026-03-01' in ref for ref in state['summary_citations'])
                assert any('sberprime-offer@2026-09-30' in ref for ref in state['summary_citations'])
                assert 'ФОН-9999' not in state['summary'] and 'ФОН-8888' not in state['summary']
                exact_id_calls = report['turns'][20]['rag_tool_calls']
                assert any(call['arguments']['source_id'] == 'sberprime-offer@2026-03-01' for call in exact_id_calls), 'Не показан поиск по идентификатору'
                assert any(turn['uncompacted_preflight_tokens'] > extractor.context.input_tokens
                           for turn in report['turns']), 'Не продемонстрирован рост истории сверх бюджета'
                calls = [call for entry in state['budget_log'] for call in entry['calls']]
                assert all(call.get('within_budget') is not False for call in calls)
                assert all(call.get('usage', {}).get('input_tokens') is not None for call in calls)
                report.update(passed=True, commands=[asdict(command) for command in software.commands],
                              session_usage=state['session_usage'], folded_messages=state['folded_messages'],
                              max_dialogue_actual_input=max(call['usage']['input_tokens'] for call in calls
                                  if call['stage'] in ('extract', 'preference', 'summarize')),
                              max_rag_actual_input=max(call['usage']['input_tokens'] for call in calls
                                  if call['stage'] in ('tool_selection', 'grounded_answer')))
            finally:
                store.close()
    finally:
        index.close()
        if hasattr(embeddings, 'close'):
            embeddings.close()
        output = DATA / 'context/long_dialogue_report.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Длинный сценарий пройден: 35 реплик, сводки, фактический бюджет и одна подтверждённая выдача.', flush=True)


if __name__ == '__main__':
    from atm_agent.config.diagnostics import connection_error

    parser = argparse.ArgumentParser(description='Проверка длинного контекста на живом GigaChat')
    parser.add_argument('--env-file')
    args = parser.parse_args()
    try:
        run(args.env_file)
    except Exception as exc:
        print(str(exc) if isinstance(exc, AssertionError) else connection_error(exc), flush=True)
        raise SystemExit(1)
