import argparse
import hashlib
import json
import logging
import re
from datetime import date
from pathlib import Path

import pypdf
from pypdf import PdfReader

from atm_agent.config.paths import PROJECT, REPOSITORY
from atm_agent.knowledge.versions import resolve_intervals

KNOWLEDGE = PROJECT / 'data' / 'knowledge'


class PDFWarnings(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def check_document(document: dict, root: Path) -> Path:
    """Проверяет источник и ручную разметку до извлечения; даты из имени не выводит."""
    path = (root / document['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Документ должен находиться внутри репозитория.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != document['sha256']:
        raise ValueError(f"Изменился PDF {document['path']}; проверьте каталог заново.")
    for key in ('revision_date', 'valid_from', 'valid_to'):
        if document.get(key) is not None:
            date.fromisoformat(document[key])
    if document.get('valid_from') and document.get('valid_to'):
        if document['valid_from'] >= document['valid_to']:
            raise ValueError('Пустой или обратный интервал действия редакции.')
    return path


def extract_document(document: dict, root: Path) -> tuple[dict, list[dict]]:
    """Сохраняет обычный и позиционный текст со ссылками на страницы исходного PDF."""
    path = check_document(document, root)
    warnings = PDFWarnings()
    logger = logging.getLogger('pypdf')
    previous_propagate = logger.propagate
    logger.propagate = False
    logger.addHandler(warnings)
    try:
        reader = PdfReader(path)
        pages = []
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ''
            layout = page.extract_text(extraction_mode='layout', layout_mode_space_vertically=False) or ''
            pages.append({
                'page_id': f"{document['source_id']}:p{number:04}",
                'source_id': document['source_id'], 'page': number,
                'text': text, 'layout_text': layout,
                'citation': f"[{document['source_id']}, с. {number}]",
                'source_path': document['path'],
            })
        evidence = document.get('date_evidence')
        if evidence:
            text = pages[evidence['page'] - 1]['text']
            if normalize(evidence['quote']) not in normalize(text):
                raise ValueError(f"Не подтверждена дата в {document['path']}.")
    finally:
        logger.removeHandler(warnings)
        logger.propagate = previous_propagate
    empty = [page['page'] for page in pages if not normalize(page['text'])]
    if empty:
        raise ValueError(f"В {document['path']} нет текста на страницах {empty}; требуется OCR.")
    metadata = {
        **document, 'page_count': len(pages),
        'text_chars': sum(len(page['text']) for page in pages),
        'parser': 'pypdf', 'parser_version': pypdf.__version__,
        'warnings': warnings.messages, 'review_status': 'extracted_not_indexed',
    }
    return metadata, pages


def write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    temporary.replace(path)


def prepare(catalog_path: Path, output: Path, root: Path = REPOSITORY) -> dict:
    """Формирует проверяемые страницы корпуса; поиск и эмбеддинги ещё не запускает."""
    catalog = json.loads(catalog_path.read_text())
    documents = resolve_intervals(catalog['documents'], catalog['temporal_policy'])
    sources, pages = [], []
    seen = set()
    for document in documents:
        if document['source_id'] in seen:
            raise ValueError('Повторяющийся source_id в каталоге.')
        seen.add(document['source_id'])
        metadata, extracted = extract_document(document, root)
        sources.append(metadata)
        pages.extend(extracted)
        print(f"{document['source_id']}: {len(extracted)} стр., предупреждений PDF: {len(metadata['warnings'])}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'documents.jsonl', sources)
    write_jsonl(output / 'pages.jsonl', pages)
    summary = {
        'corpus_version': catalog['corpus_version'],
        'temporal_policy': catalog['temporal_policy'],
        'document_count': len(sources), 'page_count': len(pages),
        'document_families': len({source['document_id'] for source in sources}),
        'undated_sources': [source['source_id'] for source in sources if not source['revision_date']],
        'marker_only_sources': [source['source_id'] for source in sources if source['date_basis'] == 'revision_marker'],
        'effective_after_inspection_date': [source['source_id'] for source in sources
                                           if source.get('valid_from') and source['valid_from'] > date.today().isoformat()],
        'inspection_date': date.today().isoformat(),
        'index_built': False,
    }
    (output / 'inventory.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Подготовка загруженных PDF для базы знаний')
    parser.add_argument('--catalog', type=Path, default=KNOWLEDGE / 'catalog.json')
    parser.add_argument('--output', type=Path, default=KNOWLEDGE)
    args = parser.parse_args()
    summary = prepare(args.catalog, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
