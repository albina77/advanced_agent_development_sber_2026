import hashlib
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from atm_agent.knowledge.documents import check_document, extract_document


class DocumentsTest(unittest.TestCase):
    def make_document(self, root):
        writer = PdfWriter()
        page = writer.add_blank_page(width=400, height=400)
        font = DictionaryObject({
            NameObject('/Type'): NameObject('/Font'),
            NameObject('/Subtype'): NameObject('/Type1'),
            NameObject('/BaseFont'): NameObject('/Helvetica'),
        })
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): font}),
        })
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 40 350 Td (Edition effective 2026-10-28) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        path = root / 'sample.pdf'
        with path.open('wb') as output:
            writer.write(output)
        return {
            'path': 'sample.pdf', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_id': 'test@2026-10-28', 'revision_date': '2026-10-28',
            'valid_from': '2026-10-28', 'valid_to': None,
            'date_evidence': {'page': 1, 'quote': 'Edition effective 2026-10-28'},
        }

    def test_date_and_page_citation_survive_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metadata, pages = extract_document(self.make_document(root), root)
        self.assertEqual(metadata['valid_from'], '2026-10-28')
        self.assertEqual(metadata['page_count'], 1)
        self.assertIn('2026-10-28', pages[0]['layout_text'])
        self.assertEqual(pages[0]['citation'], '[test@2026-10-28, с. 1]')

    def test_changed_pdf_requires_metadata_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            document = self.make_document(root)
            (root / 'sample.pdf').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Изменился PDF'):
                check_document(document, root)

    def test_date_evidence_must_exist_in_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            document = self.make_document(root)
            document['date_evidence']['quote'] = 'Edition effective 2024-01-01'
            with self.assertRaisesRegex(ValueError, 'Не подтверждена дата'):
                extract_document(document, root)

    def test_unknown_date_is_not_replaced_by_import_date(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            document = self.make_document(root)
            document.update(revision_date=None, valid_from=None, date_evidence=None)
            metadata, _ = extract_document(document, root)
        self.assertIsNone(metadata['revision_date'])
        self.assertIsNone(metadata['valid_from'])
