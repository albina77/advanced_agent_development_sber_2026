import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import httpx

from atm_agent.knowledge.collect_sources import collect


class SourcesTest(unittest.TestCase):
    def test_http_200_service_page_is_rejected_and_provenance_is_saved(self):
        html = '<html><body>Возникла проблема при открытии сайта Сбербанка в этом браузере.</body></html>'
        transport = httpx.MockTransport(lambda request: httpx.Response(
            200, text=html, headers={'content-type': 'text/html'},
        ))
        with tempfile.TemporaryDirectory() as folder:
            with patch('atm_agent.knowledge.collect_sources.DATA_DIR', Path(folder)), redirect_stdout(io.StringIO()):
                with httpx.Client(transport=transport) as client:
                    result = collect('sberkarta', 'https://www.sberbank.ru/product', client)
            self.assertFalse(result['usable'])
            self.assertEqual(result['rejection_reason'], 'service_page_instead_of_product')
            self.assertEqual(result['review_status'], 'unreviewed')
            self.assertTrue(result['fetched_at'])
            self.assertEqual(result['requested_url'], 'https://www.sberbank.ru/product')
            self.assertEqual(len(result['sha256']), 64)
            self.assertEqual(len(list(Path(folder).glob('sberkarta/*/source.json'))), 1)
