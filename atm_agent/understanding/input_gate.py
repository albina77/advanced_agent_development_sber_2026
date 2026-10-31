from enum import Enum
from typing import Callable, TypeVar


class Relevance(str, Enum):
    ADDRESSED = 'addressed'
    BACKGROUND = 'background'
    UNCERTAIN = 'uncertain'


T = TypeVar('T')


def dispatch_turn(
    text: str,
    relevance: Relevance,
    handle_addressed: Callable[[str], T],
) -> T | None:
    """Пропускает к обработчику только речь, уверенно адресованную банкомату."""
    if relevance != Relevance.ADDRESSED:
        return None
    return handle_addressed(text)
