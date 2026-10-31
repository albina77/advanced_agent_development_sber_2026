import json
from urllib.parse import urlsplit


def connection_error(exc: Exception) -> str:
    """Показывает статус, этап запроса и числовой код API, без тела и заголовков."""
    from atm_agent.retrieval.embeddings import CloudEmbeddingError

    if isinstance(exc, CloudEmbeddingError):
        return str(exc)
    details = [f'Не удалось подключить GigaChat ({type(exc).__name__}).']
    status = getattr(exc, 'status_code', None)
    if type(status) is int:
        details.append(f'HTTP {status}.')
    try:
        url = urlsplit(str(getattr(exc, 'url', '')))
        if url.hostname:
            details.append(f'Этап: {"OAuth — получение токена" if url.path.endswith("/oauth") else "запрос к API"}.')
    except ValueError:
        pass
    try:
        body = json.loads(getattr(exc, 'content', None) or '{}')
        code = body.get('code') if isinstance(body, dict) else None
    except (TypeError, ValueError, UnicodeError):
        code = None
    if type(code) is int:
        details.append(f'Код GigaChat: {code}.')
    if status == 401:
        if code == 4:
            details.append('Сервис не смог декодировать ключ авторизации. Проверьте Authorization Key в .env.')
        elif code == 6:
            details.append('Сервис не принял ключ для выбранного scope. Проверьте соответствие ключа и типа проекта; ключ мог быть перевыпущен.')
        else:
            details.append('Сервис отклонил авторизацию. Проверьте действующий ключ и scope проекта.')
    elif status == 403:
        details.append('Доступ запрещён. Проверьте права проекта на выбранную модель.')
    elif status is None:
        details.append('Проверьте сеть и настройки подключения.')
    return ' '.join(details)
