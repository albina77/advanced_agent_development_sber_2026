import argparse
import hashlib
import json
import ssl
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from atm_agent.config.paths import DATA

SEEDS = {
    'sberkarta': 'https://www.sberbank.ru/ru/person/bank_cards/debit/sberkarta',
    'deposit': 'https://www.sberbank.ru/ru/person/contributions/deposits/best',
    'savings': 'https://www.sberbank.ru/ru/person/contributions/accounts/nakopitelny',
}
SRAVNI_SEEDS = {
    'sberkarta': 'https://www.sravni.ru/debetovye-karty/bank/sberbank-rossii/',
    'deposit': 'https://www.sravni.ru/vklady/info/sber-luchshij-procent/',
    'savings': 'https://www.sravni.ru/bank/sberbank-rossii/vklady/nakopitelnye/',
}
CERTIFICATE_URL = 'https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt'
DATA_DIR = DATA / 'sources'


class PageText(HTMLParser):
    """Сохраняет видимый текст и ссылки; JavaScript не исполняет."""

    def __init__(self):
        super().__init__()
        self.text = []
        self.links = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript', 'svg'):
            self.skip += 1
        if not self.skip and tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript', 'svg'):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, text):
        if not self.skip and text.strip():
            self.text.append(' '.join(text.split()))


def sber_context() -> ssl.SSLContext:
    """Доверяет CA для этого загрузчика; системные сертификаты не меняет."""
    certificate = DATA_DIR / 'russian_trusted_root_ca_pem.crt'
    certificate.parent.mkdir(parents=True, exist_ok=True)
    if not certificate.exists():
        response = httpx.get(CERTIFICATE_URL, timeout=20)
        response.raise_for_status()
        ssl.PEM_cert_to_DER_cert(response.text)
        certificate.write_bytes(response.content)
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=str(certificate))
    return context


def collect(name: str, url: str, client: httpx.Client) -> dict:
    """Сохраняет снимок источника и происхождение; дату загрузки не считает датой редакции."""
    response = client.get(url)
    fetched_at = datetime.now(timezone.utc).isoformat()
    digest = hashlib.sha256(response.content).hexdigest()
    page = PageText()
    if 'html' in response.headers.get('content-type', ''):
        page.feed(response.text)
    links = sorted({urljoin(str(response.url), href) for href in page.links})
    result = {
        'source_id': name, 'requested_url': url, 'final_url': str(response.url),
        'fetched_at': fetched_at, 'http_status': response.status_code,
        'content_type': response.headers.get('content-type'), 'sha256': digest,
        'text': '\n'.join(page.text), 'links': links,
    }
    result['usable'] = response.status_code == 200 and len(result['text']) > 1000
    result['review_status'] = 'unreviewed'
    if 'Возникла проблема при открытии сайта' in result['text']:
        result['usable'] = False
        result['rejection_reason'] = 'service_page_instead_of_product'
    folder = DATA_DIR / name / digest[:12]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'response.bin').write_bytes(response.content)
    (folder / 'source.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({
        'source_id': name, 'status': response.status_code, 'bytes': len(response.content),
        'text_chars': len(result['text']), 'preview': result['text'][:350],
        'usable': result['usable'],
        'pdf_links': [link for link in links if '.pdf' in link.lower()][:30],
        'snapshot': str(folder),
    }, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    """Загружает снимки выбранных источников и сохраняет результат проверки каждого адреса."""
    parser = argparse.ArgumentParser(description='Снимки источников по продуктам Сбера для учебного RAG')
    parser.add_argument('--product', choices=SEEDS, action='append')
    parser.add_argument('--source', choices=('sber', 'sravni'), default='sber')
    parser.add_argument('--domain', choices=('www.sberbank.ru', 'www.sberbank.com'), default='www.sberbank.ru')
    args = parser.parse_args()
    try:
        context = sber_context()
        seeds = SEEDS if args.source == 'sber' else SRAVNI_SEEDS
        results = []
        with httpx.Client(verify=context, follow_redirects=True, timeout=25) as client:
            for name in args.product or seeds:
                url = seeds[name]
                if args.source == 'sber':
                    url = urlsplit(url)._replace(netloc=args.domain).geturl()
                results.append(collect(f'{args.source}-{name}', url, client))
        if not any(result['usable'] for result in results):
            print('Пригодные страницы не получены. Сохранены только диагностические снимки.')
            return 1
    except (httpx.HTTPError, ValueError, ssl.SSLError) as exc:
        print(f'Не удалось загрузить источник: {type(exc).__name__}: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
