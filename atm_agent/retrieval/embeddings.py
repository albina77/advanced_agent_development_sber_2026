import math

import httpx


class CloudEmbeddingError(RuntimeError):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f'Cloud.ru embeddings: HTTP {status_code}')


class CloudEmbeddings:
    """Адаптер embeddings API Cloud.ru с раздельной обработкой документов и поисковых запросов."""
    use_prefix_query = True
    prefix_query = 'Instruct: Дан вопрос, необходимо найти абзац текста с ответом\nQuery: '

    def __init__(self, api_key, model='ai-sage/Giga-Embeddings-instruct-3B', transport=None):
        self.remote_model = model
        self.model = 'cloud_ru/' + model
        self.usage = {'requests': 0, 'prompt_tokens': 0, 'total_tokens': 0}
        self.client = httpx.Client(
            base_url='https://foundation-models.api.cloud.ru/v1/', timeout=120,
            headers={'Authorization': 'Bearer ' + api_key}, transport=transport,
        )

    def embed_documents(self, texts):
        """Запрашивает векторы и проверяет порядок, полноту и корректность ответа без вывода тела ошибки."""
        if not texts:
            return []
        response = self.client.post('embeddings', json={
            'model': self.remote_model, 'input': texts, 'encoding_format': 'float',
        })
        if not response.is_success:
            raise CloudEmbeddingError(response.status_code)
        body = response.json()
        items = body.get('data', [])
        indices = [item.get('index') for item in items]
        if len(items) != len(texts) or set(indices) != set(range(len(texts))):
            raise ValueError('Cloud.ru вернул неполный или повторяющийся набор векторов.')
        vectors = [item['embedding'] for item in sorted(items, key=lambda item: item['index'])]
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or 0 in dimensions or any(
            not all(type(value) in (int, float) and math.isfinite(value) for value in vector)
            or not any(vector) for vector in vectors
        ):
            raise ValueError('Cloud.ru вернул некорректные векторы.')
        self.usage['requests'] += 1
        for key in ('prompt_tokens', 'total_tokens'):
            self.usage[key] += body.get('usage', {}).get(key, 0)
        return vectors

    def embed_query(self, text):
        return self.embed_documents([self.prefix_query + text])[0]

    def close(self):
        self.client.close()
