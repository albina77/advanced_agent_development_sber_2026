import unittest

from atm_agent.config.diagnostics import connection_error


class DiagnosticsTest(unittest.TestCase):
    def test_cloud_error_identifies_provider(self):
        from atm_agent.retrieval.embeddings import CloudEmbeddingError

        self.assertEqual(connection_error(CloudEmbeddingError(402)), 'Cloud.ru embeddings: HTTP 402')

    def test_auth_failure_explains_code_without_echoing_response(self):
        error = RuntimeError('secret-in-exception')
        error.status_code = 401
        error.url = 'https://user:secret-in-url@example.org/api/v2/oauth?token=secret-in-query'
        error.content = b'{"code": 6, "message": "secret-in-response"}'
        error.headers = {'Authorization': 'secret-in-header'}
        text = connection_error(error)
        self.assertIn('HTTP 401', text)
        self.assertIn('OAuth', text)
        self.assertIn('scope', text)
        self.assertNotIn('secret', text)

    def test_network_failure_is_not_reported_as_rejected_credentials(self):
        text = connection_error(ConnectionError('secret-in-exception'))
        self.assertIn('сеть', text)
        self.assertNotIn('отклонил авторизацию', text)
        self.assertNotIn('secret', text)
