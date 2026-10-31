import json
import unittest

from atm_agent.knowledge.versions import (
    CATALOG,
    TEMPORAL_POLICY,
    resolve_intervals,
    select_sources,
)


class DocumentVersionsTest(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads(CATALOG.read_text())
        self.documents = resolve_intervals(self.catalog['documents'], self.catalog['temporal_policy'])

    def versions(self, family, day):
        return [entry['version'] for entry in select_sources(self.documents, day, document_id=family)]

    def test_spasibo_switches_on_effective_day(self):
        self.assertEqual(self.versions('sberspasibo-rules', '2026-10-27'), ['2024-10-01'])
        self.assertEqual(self.versions('sberspasibo-rules', '2026-10-28'), ['2026-10-28'])
        self.assertEqual(self.versions('sberspasibo-rules', '2024-09-30'), ['2024-07-01'])
        self.assertEqual(self.versions('sberspasibo-rules', '2024-10-01'), ['2024-10-01'])

    def test_offer_transitions_and_dates_before_first_revision(self):
        self.assertEqual(self.versions('sberprime-offer', '2025-10-21'), [])
        self.assertEqual(self.versions('sberprime-offer', '2026-02-28'), ['2025-10-22'])
        self.assertEqual(self.versions('sberprime-offer', '2026-03-01'), ['2026-03-01'])
        self.assertEqual(self.versions('sberprime-offer', '2026-09-30'), ['2026-09-30'])

    def test_partner_marker_assumption_is_explicit(self):
        entry = select_sources(self.documents, '2026-10-27', document_id='sberprime-services')[0]
        self.assertEqual(entry['version'], '2026-10-27')
        self.assertEqual(entry['valid_from_basis'], 'revision_marker_assumed')
        self.assertIsNone(entry['document_valid_from'])

    def test_tariff_period_does_not_backdate_revision_or_extend_explicit_end(self):
        self.assertEqual(self.versions('sberprime-bank-tariffs', '2023-10-02'), [])
        self.assertEqual(self.versions('sberprime-bank-tariffs', '2026-08-10'), [])
        self.assertEqual(self.versions('sberprime-bank-tariffs', '2026-08-11'), ['2026-08-11'])
        self.assertEqual(self.versions('sberprime-bank-tariffs', '2027-03-17'), ['2026-08-11'])
        self.assertEqual(self.versions('sberprime-bank-tariffs', '2027-03-18'), [])

    def test_undated_card_is_excluded_from_date_search(self):
        self.assertEqual(select_sources(self.documents, '2026-10-28', product='sberkarta'), [])

    def test_intervals_are_separate_for_each_document_family(self):
        found = select_sources(self.documents, '2026-10-28', product='sberprime')
        self.assertEqual({entry['document_id'] for entry in found}, {
            'sberprime-offer', 'sberprime-services', 'sberprime-bank-tariffs',
        })
        self.assertEqual(len(found), 3)

    def test_catalog_facts_are_not_overwritten_by_inferred_intervals(self):
        original = next(entry for entry in self.catalog['documents']
                        if entry['source_id'] == 'sberprime-offer@2025-10-22')
        derived = next(entry for entry in self.documents if entry['source_id'] == original['source_id'])
        self.assertIsNone(original['valid_to'])
        self.assertEqual(derived['valid_to'], '2026-03-01')
        self.assertEqual(derived['valid_to_basis'], 'next_loaded_revision')

    def test_duplicate_dates_are_rejected(self):
        original = self.catalog['documents'][1]
        with self.assertRaisesRegex(ValueError, 'одинаковая дата'):
            resolve_intervals([original, dict(original, source_id='duplicate')], TEMPORAL_POLICY)

    def test_explicit_earlier_end_is_preserved(self):
        first = dict(self.catalog['documents'][1], valid_to='2024-08-01')
        second = self.catalog['documents'][2]
        resolved = resolve_intervals([first, second], TEMPORAL_POLICY)
        self.assertEqual(resolved[0]['valid_to'], '2024-08-01')
        self.assertEqual(select_sources(resolved, '2024-09-01'), [])
