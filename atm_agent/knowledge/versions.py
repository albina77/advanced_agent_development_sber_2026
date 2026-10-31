import argparse
import json
from collections import defaultdict
from datetime import date

from atm_agent.config.paths import DATA

TEMPORAL_POLICY = 'until_next_loaded_revision'
CATALOG = DATA / 'knowledge' / 'catalog.json'


def parse_date(value: str) -> date:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError('Нужна дата в формате YYYY-MM-DD.')
    return parsed


def resolve_intervals(documents: list[dict], policy: str) -> list[dict]:
    """Выводит интервалы внутри каждого семейства по согласованному учебному допущению."""
    if policy != TEMPORAL_POLICY:
        raise ValueError('Не согласована политика интервалов редакций.')
    resolved = []
    families = defaultdict(list)
    for document in documents:
        entry = dict(document)
        start, end = document.get('valid_from'), document.get('valid_to')
        entry.update(document_valid_from=start, document_valid_to=end)
        entry['valid_from_basis'] = 'explicit_document' if start else 'unknown'
        if start is None and document.get('date_basis') == 'revision_marker':
            start = document.get('revision_date')
            entry['valid_from_basis'] = 'revision_marker_assumed'
        entry['valid_from'] = start
        entry['valid_to_basis'] = 'explicit_document' if end else 'open_ended'
        if start:
            parse_date(start)
            if end and parse_date(end) <= parse_date(start):
                raise ValueError('Некорректный интервал действия редакции.')
            families[entry['document_id']].append(entry)
        elif end:
            raise ValueError('Задано окончание действия без даты начала.')
        resolved.append(entry)
    for revisions in families.values():
        revisions.sort(key=lambda entry: entry['valid_from'])
        for current, following in zip(revisions, revisions[1:]):
            boundary = following['valid_from']
            if current['valid_from'] == boundary:
                raise ValueError('У двух редакций одного документа одинаковая дата начала.')
            if current['valid_to'] is None or boundary < current['valid_to']:
                current['valid_to'] = boundary
                current['valid_to_basis'] = 'next_loaded_revision'
    return resolved


def select_sources(
    documents: list[dict], as_of: str,
    document_id: str | None = None, product: str | None = None,
) -> list[dict]:
    """Выбирает источники по интервалу [valid_from, valid_to), исключая недатированные."""
    target = parse_date(as_of)
    selected = []
    for document in documents:
        if document_id and document['document_id'] != document_id:
            continue
        if product and document['product'] != product:
            continue
        start, end = document.get('valid_from'), document.get('valid_to')
        if start is None or target < parse_date(start):
            continue
        if end and target >= parse_date(end):
            continue
        selected.append(document)
    return selected


def main() -> None:
    """Показывает редакции, действующие на указанную дату с учётом фильтров документа."""
    parser = argparse.ArgumentParser(description='Выбор редакции по дате внутри учебного корпуса')
    parser.add_argument('--as-of', required=True)
    parser.add_argument('--document-id')
    parser.add_argument('--product')
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text())
    documents = resolve_intervals(catalog['documents'], catalog['temporal_policy'])
    selected = select_sources(documents, args.as_of, args.document_id, args.product)
    print(json.dumps({
        'as_of': args.as_of, 'policy': catalog['temporal_policy'],
        'sources': [{key: entry.get(key) for key in (
            'source_id', 'path', 'valid_from', 'valid_to', 'valid_from_basis', 'valid_to_basis',
        )} for entry in selected],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
