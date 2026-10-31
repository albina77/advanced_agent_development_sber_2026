import os
from pathlib import Path

from atm_agent.config.paths import PROJECT


def load_settings(env_file: str | None = None) -> None:
    """Загружает указанный или первый найденный .env, сохраняя приоритет переменных окружения."""
    from dotenv import load_dotenv

    project = PROJECT
    if env_file:
        source = Path(env_file).expanduser().resolve()
        if not source.is_file():
            raise ValueError('Указанный файл настроек не найден.')
        load_dotenv(source, override=False)
    else:
        for source in (
            project / '.env',
            project.parent / '.env',
            project.parent.parent / 'prompting_and_tokenization_sber_2026' / '.env',
        ):
            if source.is_file():
                load_dotenv(source, override=False)
                break


def connection_options(env_file: str | None = None) -> dict:
    """Проверяет настройки и собирает параметры подключения GigaChat без вывода ключей."""
    load_settings(env_file)
    if not os.getenv('GIGACHAT_CREDENTIALS'):
        raise ValueError('Задайте GIGACHAT_CREDENTIALS или укажите --env-file.')
    verify = os.getenv('GIGACHAT_VERIFY_SSL_CERTS', 'true').lower()
    if verify not in ('true', 'false', '1', '0'):
        raise ValueError('GIGACHAT_VERIFY_SSL_CERTS должен быть true или false.')
    options = {
        'credentials': os.environ['GIGACHAT_CREDENTIALS'],
        'scope': os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_B2B'),
        'verify_ssl_certs': verify in ('true', '1'),
        'timeout': 60,
    }
    for variable, option in (
        ('GIGACHAT_BASE_URL', 'base_url'),
        ('GIGACHAT_AUTH_URL', 'auth_url'),
        ('GIGACHAT_CA_BUNDLE_FILE', 'ca_bundle_file'),
    ):
        if os.getenv(variable):
            options[option] = os.environ[variable]
    return options


def create_model(env_file: str | None = None):
    from langchain_gigachat import GigaChat

    return GigaChat(**connection_options(env_file),
                    model=os.getenv('GIGACHAT_MODEL_NAME', 'GigaChat-2-Max'), temperature=0,
                    max_tokens=int(os.getenv('MODEL_MAX_OUTPUT_TOKENS', '1024')))


def create_embeddings(env_file: str | None = None):
    """Создаёт эмбеддер выбранного провайдера; по умолчанию наличие ключа Cloud.ru выбирает его."""
    load_settings(env_file)
    cloud_key = os.getenv('CLOUD_RU_CREDENTIALS') or os.getenv('cloud_ru_credentials')
    provider = os.getenv('EMBEDDING_PROVIDER', 'cloud_ru' if cloud_key else 'gigachat')
    if provider == 'cloud_ru':
        from atm_agent.retrieval.embeddings import CloudEmbeddings

        if not cloud_key:
            raise ValueError('Задайте cloud_ru_credentials в .env.')
        return CloudEmbeddings(cloud_key,
                               model=os.getenv('CLOUD_RU_EMBEDDING_MODEL', 'ai-sage/Giga-Embeddings-instruct-3B'))
    if provider != 'gigachat':
        raise ValueError('EMBEDDING_PROVIDER должен быть cloud_ru или gigachat.')
    from langchain_gigachat import GigaChatEmbeddings

    return GigaChatEmbeddings(**connection_options(env_file),
                             model=os.getenv('GIGACHAT_EMBEDDING_MODEL', 'EmbeddingsGigaR'),
                             use_prefix_query=True,
                             prefix_query='Дан вопрос, необходимо найти абзац корпоративного документа с ответом.\nвопрос:\n')
