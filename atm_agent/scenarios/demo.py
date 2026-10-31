from atm_agent.atm.contracts import ATMCommand
from atm_agent.atm.mock import DemoATMSoftware
from atm_agent.dialogue.state import Turn, handle_turn, receive_result
from atm_agent.understanding.input_gate import Relevance


def main():
    """Показывает накопление параметров и подтверждение на заданных намерениях без вызова модели."""
    print('Локальный пример: извлечение задано вручную, без GigaChat и LangGraph.')
    software = DemoATMSoftware({})
    state = {}
    turns = [
        ('Хочу снять', Turn(Relevance.ADDRESSED, 'withdraw')),
        ('Семь тысяч', Turn(Relevance.ADDRESSED, 'details', amount=7000)),
        ('Объявляется посадка на рейс', Turn(Relevance.BACKGROUND)),
        ('Тысячными', Turn(Relevance.ADDRESSED, 'details', denomination=1000)),
        ('Да', Turn(Relevance.ADDRESSED, 'confirm')),
    ]
    for text, turn in turns:
        print('Вы:', text)
        state.update(handle_turn(state, turn))
        if state.get('outgoing'):
            result = software.send(ATMCommand(**state['outgoing']))
            state.update(receive_result(state, result))
        if state.get('response'):
            print('Агент:', state['response'])
    print('Команд отправлено:', len(software.commands))
    print('Команда:', software.commands[0])


if __name__ == '__main__':
    main()
