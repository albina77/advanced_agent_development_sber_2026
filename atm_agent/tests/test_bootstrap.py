import unittest
from unittest.mock import Mock, patch

from atm_agent.knowledge.bootstrap import ensure_ready


class BootstrapTests(unittest.TestCase):
    def run_bootstrap(self, ready, load_error=None):
        embeddings, reranker = Mock(), Mock()
        reranker.load.side_effect = [load_error, None] if load_error else None
        with (patch('atm_agent.knowledge.bootstrap.create_embeddings', return_value=embeddings),
              patch('atm_agent.knowledge.bootstrap.prepare_sources') as sources,
              patch('atm_agent.knowledge.bootstrap.LocalReranker', return_value=reranker),
              patch('atm_agent.knowledge.bootstrap.prepare_reranker') as prepare_model,
              patch('atm_agent.knowledge.bootstrap.index_ready', return_value=ready),
              patch('atm_agent.knowledge.bootstrap.build') as build):
            result = ensure_ready('custom.env')
        return result, embeddings, reranker, sources, prepare_model, build

    def test_ready_index_does_not_request_new_embeddings(self):
        result, embeddings, reranker, sources, prepare_model, build = self.run_bootstrap(True)
        self.assertEqual(result, (embeddings, reranker))
        sources.assert_called_once_with()
        prepare_model.assert_not_called()
        build.assert_not_called()

    def test_clean_start_prepares_model_and_index(self):
        _, embeddings, reranker, _, prepare_model, build = self.run_bootstrap(False, OSError())
        prepare_model.assert_called_once_with()
        self.assertEqual(reranker.load.call_count, 2)
        build.assert_called_once_with(embeddings)

    def test_missing_model_does_not_rebuild_ready_vectors(self):
        *_, prepare_model, build = self.run_bootstrap(True, OSError())
        prepare_model.assert_called_once_with()
        build.assert_not_called()
