import json
import unittest

import httpx

from atm_agent.retrieval.embeddings import CloudEmbeddingError, CloudEmbeddings


class CloudEmbeddingsTest(unittest.TestCase):
    def test_query_prefix_and_response_order(self):
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            count = len(payload['input'])
            return httpx.Response(200, json={
                'data': [{'index': index, 'embedding': [index + 1., 0.5]}
                         for index in reversed(range(count))],
                'usage': {'prompt_tokens': 10, 'total_tokens': 10},
            })

        client = CloudEmbeddings('test-key', transport=httpx.MockTransport(respond))
        self.addCleanup(client.close)
        self.assertEqual(client.embed_documents(['one', 'two']), [[1., 0.5], [2., 0.5]])
        client.embed_query('Бонусы?')
        self.assertEqual(requests[0]['input'], ['one', 'two'])
        self.assertEqual(requests[1]['input'], [client.prefix_query + 'Бонусы?'])
        self.assertEqual(client.usage['total_tokens'], 20)

    def test_rejects_invalid_vectors_and_indices(self):
        for data in ([{'index': 0, 'embedding': [0., 0.]}],
                     [{'index': 1, 'embedding': [1., 2.]}],
                     [{'index': 0, 'embedding': [float('inf'), 1.]}], []):
            with self.subTest(data=data):
                transport = httpx.MockTransport(lambda request: httpx.Response(
                    200, content=json.dumps({'data': data})))
                client = CloudEmbeddings('test-key', transport=transport)
                self.addCleanup(client.close)
                with self.assertRaises(ValueError):
                    client.embed_documents(['one'])

    def test_error_hides_response_and_credentials(self):
        client = CloudEmbeddings('secret-key', transport=httpx.MockTransport(
            lambda request: httpx.Response(401, text='secret response')))
        self.addCleanup(client.close)
        with self.assertRaises(CloudEmbeddingError) as raised:
            client.embed_query('test')
        self.assertEqual(str(raised.exception), 'Cloud.ru embeddings: HTTP 401')


if __name__ == '__main__':
    unittest.main()
