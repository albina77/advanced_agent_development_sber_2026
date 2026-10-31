import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from langchain_core.messages import AIMessage

from atm_agent.knowledge.corpus import split_page
from atm_agent.retrieval.consultant import (
    Claim,
    Evidence,
    GroundedAnswer,
    RAGConsultant,
    RejectedClaim,
    SupportReview,
    overview_options,
    overview_question,
    quote_options,
    render_answer,
    undated_conditions,
    validate_answer,
)
from atm_agent.retrieval.index import HybridIndex, rrf


class FakeEmbeddings:
    model = 'test-embeddings'

    def __init__(self):
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        return [1.0, 0.0]


class FakeReranker:
    def rank(self, query, candidates):
        return [(item['chunk_id'], float(len(candidates) - index)) for index, item in enumerate(candidates)]


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.embeddings = FakeEmbeddings()
        chunks, documents = [], []
        for name, start, end in [('old', '2024-01-01', '2026-01-01'), ('new', '2026-01-01', None)]:
            documents.append({'source_id': name, 'document_id': 'rules', 'product': 'sberspasibo',
                              'valid_from': start, 'valid_to': end})
            chunks.append({'chunk_id': name + ':p1:c0', 'source_id': name, 'document_id': 'rules',
                           'product': 'sberspasibo', 'text': 'Бонусы начисляются за покупки.',
                           'index_text': 'Правила программы: бонусы начисляются за покупки.',
                           'source_path': 'docs/rules.pdf', 'page': 1})
        (self.root / 'index.json').write_text(json.dumps({'embedding_model': self.embeddings.model}))
        for filename, values in [('chunks', chunks), ('documents', documents)]:
            (self.root / f'{filename}.jsonl').write_text('\n'.join(json.dumps(item) for item in values))
        np.save(self.root / 'vectors.npy', np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32))
        self.index = HybridIndex(self.embeddings, FakeReranker(), self.root, validate=False)

    def tearDown(self):
        self.index.close()
        self.folder.cleanup()

    def test_both_legs_exclude_old_revision_before_ranking(self):
        result = self.index.search('начисление бонусов', '2026-01-01')
        self.assertEqual([item['source_id'] for item in result['chunks']], ['new'])
        for stage in ('lexical', 'dense', 'rrf', 'reranked'):
            self.assertTrue(result['trace'][stage])
            self.assertTrue(all(item['chunk_id'].startswith('new:') for item in result['trace'][stage]))

    def test_same_query_on_different_dates_reuses_embedding(self):
        old = self.index.search('начисление бонусов', '2025-12-31')
        new = self.index.search('начисление бонусов', '2026-01-01')
        self.assertEqual(old['chunks'][0]['source_id'], 'old')
        self.assertEqual(new['chunks'][0]['source_id'], 'new')
        self.assertEqual(self.embeddings.calls, 1)

    def test_empty_date_pool_does_not_call_embedding_api(self):
        result = self.index.search('бонусы', '2023-01-01')
        self.assertEqual(result['chunks'], [])
        self.assertEqual(self.embeddings.calls, 0)

    def test_exact_source_filter_is_intersected_with_date_filter(self):
        result = self.index.search('бонусы', '2026-01-01', source_id='old')
        self.assertEqual(result['chunks'], [])

    def test_undated_sources_require_explicit_overview_mode(self):
        self.index.documents[0]['valid_from'] = None
        dated = self.index.search('бонусы', '2026-01-01')
        overview = self.index.search('бонусы', '2026-01-01', include_undated=True)
        self.assertEqual(dated['trace']['allowed_sources'], ['new'])
        self.assertEqual(set(overview['trace']['allowed_sources']), {'old', 'new'})
        self.assertEqual(overview['trace']['undated_sources'], ['old'])


class GroundingTest(unittest.TestCase):
    def setUp(self):
        self.chunk = {'chunk_id': 'rules:p1:c0', 'source_id': 'rules', 'source_path': 'docs/rules.pdf',
                      'page': 1, 'text': 'Бонусы начисляются за покупки.', 'index_text': 'Бонусы начисляются за покупки.'}

    def test_fabricated_source_and_fabricated_quote_are_rejected(self):
        for chunk_id, quote in [('invented', self.chunk['text']), ('rules:p1:c0', 'Начисляется сто процентов')]:
            answer = GroundedAnswer(supported=True, claims=[Claim(text='Бонусы за покупки', evidence=[
                Evidence(chunk_id=chunk_id, quote=quote),
            ])])
            with self.subTest(chunk_id=chunk_id), self.assertRaises(ValueError):
                validate_answer(answer, [self.chunk])

    def test_missing_variant_produces_clarifying_question(self):
        answer = GroundedAnswer(supported=False, clarification='Какая версия подписки вас интересует?')
        response, citations = render_answer(answer, [])
        self.assertEqual(response, answer.clarification)
        self.assertEqual(citations, [])

    def test_quote_id_resolves_to_exact_source_text(self):
        chunk = {**self.chunk, 'quote_options': quote_options(self.chunk['text'])}
        answer = GroundedAnswer(supported=True, claims=[Claim(text='Бонусы за покупки.', evidence=[
            Evidence(chunk_id=chunk['chunk_id'], quote_id='q0'),
        ])])
        _, citations = render_answer(answer, [chunk])
        self.assertEqual(citations[0]['quote'], self.chunk['text'])
        answer.claims[0].evidence[0].quote_id = 'q999'
        with self.assertRaises(ValueError):
            validate_answer(answer, [chunk])

    def test_undated_overview_cannot_include_tariff_even_if_model_approves(self):
        chunks = {self.chunk['chunk_id']: self.chunk}
        evidence = Evidence(chunk_id=self.chunk['chunk_id'], quote=self.chunk['text'])
        self.assertTrue(undated_conditions(Claim(text='Обслуживание — 150 рублей.', evidence=[evidence]),
                                           chunks, {'rules'}))
        self.assertFalse(undated_conditions(Claim(text='Это дебетовая карта.', evidence=[evidence]),
                                            chunks, {'rules'}))

    def test_overview_policy_excludes_prices_dates_and_current_conditions(self):
        self.assertTrue(overview_question('Что такое СберКарта?'))
        for question in ('Что такое СберКарта на 2024 год?', 'Расскажи про актуальные тарифы СберКарты',
                         'Что такое СберКарта и сколько стоит обслуживание?'):
            self.assertFalse(overview_question(question))

    def test_overview_context_excludes_tariff_table_before_generation(self):
        text = ('Контактный центр: 900\nТарифы по дебетовым картам\nМИР СберКарта\n'
                'Обслуживание карты 150 рублей\nПри покупках по карте бесплатно\n'
                'Лимиты снятия по карте\nОсновные и дополнительные дебетовые карты')
        options = overview_options(text)
        self.assertIn('МИР СберКарта', options.values())
        self.assertIn('Тарифы по дебетовым картам', options.values())
        self.assertNotIn('150', ' '.join(options.values()))
        self.assertNotIn('бесплатно', ' '.join(options.values()))
        self.assertTrue(all(quote in text for quote in options.values()))

    def test_review_removes_claim_with_unsupported_percentage(self):
        chunk = self.chunk

        class Model:
            model = 'fake-model'

            def bind_tools(self, *args, **kwargs):
                return self

            def invoke(self, messages):
                return AIMessage('', tool_calls=[{'name': 'search_knowledge', 'args': {
                    'query': 'бонусы',
                }, 'id': 'search', 'type': 'tool_call'}])

            def with_structured_output(self, schema, **kwargs):
                class Generator:
                    def invoke(self, messages):
                        if schema is SupportReview:
                            parsed = SupportReview(rejected=[RejectedClaim(index=1, reason='Процент отсутствует в источнике')])
                        else:
                            parsed = GroundedAnswer(supported=True, claims=[
                                Claim(text=text, evidence=[Evidence(chunk_id=chunk['chunk_id'], quote=chunk['text'])])
                                for text in ('Бонусы за покупки.', 'Бонусы составляют сто процентов.')])
                        return {'raw': AIMessage(''), 'parsed': parsed, 'parsing_error': None}
                return Generator()

        class Index:
            documents = []

            def search(self, *args, **kwargs):
                return {'chunks': [chunk], 'trace': {}}

        with tempfile.TemporaryDirectory() as folder, patch('atm_agent.retrieval.consultant.INDEX', Path(folder)):
            result = RAGConsultant(Model(), Index()).consult('Какие бонусы?')
        self.assertTrue(result['supported'])
        self.assertIn('Бонусы за покупки', result['response'])
        self.assertNotIn('сто процентов', result['response'])
        self.assertEqual(result['review_rejections'][0]['index'], 1)
        self.assertEqual(result['rag_calls'][-1]['stage'], 'answer_verification')

    def test_truncated_generation_is_not_reported_as_missing_knowledge(self):
        chunk = self.chunk

        class Model:
            model = 'fake-model'

            def bind_tools(self, *args, **kwargs):
                return self

            def invoke(self, messages):
                return AIMessage('', tool_calls=[{'name': 'search_knowledge', 'args': {
                    'query': 'бонусы', 'include_undated': True,
                }, 'id': 'search', 'type': 'tool_call'}])

            def with_structured_output(self, *args, **kwargs):
                class Generator:
                    def invoke(self, messages):
                        return {'raw': AIMessage('', response_metadata={'finish_reason': 'length'}),
                                'parsed': None, 'parsing_error': ValueError('truncated')}
                return Generator()

        class Index:
            documents = []

            def search(self, *args, **kwargs):
                self.arguments, self.options = args, kwargs
                return {'chunks': [chunk], 'trace': {}}

        index = Index()
        with tempfile.TemporaryDirectory() as folder, patch('atm_agent.retrieval.consultant.INDEX', Path(folder)):
            result = RAGConsultant(Model(), index).consult('Бонусы?', as_of='2026-10-28')
        self.assertFalse(result['supported'])
        self.assertIn('Не удалось сформировать', result['response'])
        self.assertNotIn('недостаточно информации', result['response'])
        self.assertEqual(len(result['answer_errors']), 2)
        self.assertEqual(index.arguments[1], '2026-10-28')
        self.assertEqual(index.options, {})

    def test_plain_answer_without_tool_call_is_not_used(self):
        class Model:
            model = 'fake-model'

            def bind_tools(self, *args, **kwargs):
                return self

            def invoke(self, messages):
                return AIMessage('Ставка сто процентов!')

        class Index:
            documents = []

        with tempfile.TemporaryDirectory() as folder, patch('atm_agent.retrieval.consultant.INDEX', Path(folder)):
            result = RAGConsultant(Model(), Index()).consult('Какая ставка?')
        self.assertFalse(result['supported'])
        self.assertNotIn('сто процентов', result['response'])

    def test_model_selected_search_is_executed_and_cited(self):
        chunk = self.chunk

        class Model:
            model = 'fake-model'

            def bind_tools(self, *args, **kwargs):
                return self

            def invoke(self, messages):
                return AIMessage('', tool_calls=[{'name': 'search_knowledge', 'args': {
                    'query': 'бонусы за покупки', 'as_of': '2026-10-28', 'product': 'sberspasibo',
                }, 'id': 'call-1', 'type': 'tool_call'}])

            def with_structured_output(self, schema, **kwargs):
                class Generator:
                    def invoke(self, messages):
                        if schema is SupportReview:
                            return {'raw': AIMessage(''), 'parsed': SupportReview(), 'parsing_error': None}
                        payload = json.loads(next(message.content for message in messages if message.type == 'tool'))
                        reference = payload['chunks'][0]
                        return {'raw': AIMessage(''), 'parsed': GroundedAnswer(supported=True, claims=[
                            Claim(text='Бонусы начисляются за покупки.', evidence=[
                                Evidence(chunk_id=reference['chunk_id'], quote_id=next(iter(reference['quote_options']))),
                            ]),
                        ]), 'parsing_error': None}
                return Generator()

        class Index:
            documents = []

            def __init__(self):
                self.calls = []

            def search(self, *args):
                self.calls.append(args)
                return {'chunks': [chunk], 'trace': {'selected': [chunk['chunk_id']]}}

        index = Index()
        with tempfile.TemporaryDirectory() as folder, patch('atm_agent.retrieval.consultant.INDEX', Path(folder)):
            result = RAGConsultant(Model(), index).consult('За что начисляют бонусы?')
        self.assertEqual(len(index.calls), 1)
        self.assertEqual(index.calls[0][1], '2026-10-28')
        self.assertTrue(result['supported'])
        self.assertEqual(result['citations'][0]['chunk_id'], chunk['chunk_id'])


class ChunkingTest(unittest.TestCase):
    def test_long_page_is_bounded_and_content_is_not_lost(self):
        text = '\n'.join(f'Пункт {index}: условия подписки и бонусов.' for index in range(100))
        chunks = split_page(text, max_chars=400, overlap_chars=50)
        self.assertTrue(all(len(chunk) <= 400 for chunk in chunks))
        self.assertTrue(all(line in '\n'.join(chunks) for line in text.splitlines()))

    def test_rrf_combines_rankings_without_comparing_score_scales(self):
        result = rrf([[(1, 1000), (2, 200)], [(2, 0.9), (3, 0.8)]])
        self.assertEqual(result[0][0], 2)
