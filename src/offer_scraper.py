"""Scrape offer preview URLs to extract descriptions and keywords."""

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.offer_catalog import Offer, OfferCatalog

DEFAULT_CACHE_PATH = Path('data/offer_descriptions.json')
REQUEST_TIMEOUT = 10
DELAY_BETWEEN_REQUESTS = 0.5


def scrape_offer_page(url: str) -> dict:
    """Fetch a preview URL and extract title, description, and keywords."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36'
        }, allow_redirects=True)
        resp.raise_for_status()
    except Exception:
        return {}

    soup = BeautifulSoup(resp.text, 'html.parser')

    title = ''
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    description = ''
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        description = meta_desc['content'].strip()

    og_desc = ''
    og_tag = soup.find('meta', attrs={'property': 'og:description'})
    if og_tag and og_tag.get('content'):
        og_desc = og_tag['content'].strip()

    og_title = ''
    og_title_tag = soup.find('meta', attrs={'property': 'og:title'})
    if og_title_tag and og_title_tag.get('content'):
        og_title = og_title_tag['content'].strip()

    headings = []
    for tag in ('h1', 'h2'):
        for el in soup.find_all(tag, limit=3):
            text = el.get_text(strip=True)
            if text:
                headings.append(text)

    # Build combined description
    parts = [p for p in [title, og_title, description, og_desc] + headings if p]
    combined = ' | '.join(parts)

    # Extract keywords from combined text
    keywords = extract_keywords(combined)

    return {
        'title': title,
        'og_title': og_title,
        'description': description or og_desc,
        'headings': headings,
        'keywords': keywords,
    }


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text."""
    import re
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'it', 'its', 'you', 'your', 'we', 'our', 'they', 'their', 'my',
        'not', 'no', 'all', 'each', 'every', 'any', 'some', 'more', 'most',
        'other', 'into', 'up', 'out', 'about', 'than', 'just', 'also',
        'get', 'how', 'what', 'when', 'where', 'who', 'which', 'why',
        'so', 'if', 'then', 'as', 'only', 'very', 'too', 'here', 'there',
        'now', 'new', 'one', 'two', 'first', 'last', 'best', 'free', 'us',
    }
    words = re.findall(r'[a-z]+', text.lower())
    keywords = [w for w in words if len(w) > 2 and w not in stopwords]
    # Deduplicate preserving order
    seen = set()
    unique = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


def scrape_all_offers(catalog: OfferCatalog, cache_path: Path = DEFAULT_CACHE_PATH) -> dict:
    """Scrape all offers, using cache if available. Returns the scraped data dict."""
    # Load existing cache
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

    offers = catalog.get_all_offers()
    total = len(offers)
    new_scrapes = 0

    for i, offer in enumerate(offers):
        offer_id_str = str(offer.id)
        if offer_id_str in cache:
            continue

        if not offer.preview_url or offer.preview_url.strip() == '':
            cache[offer_id_str] = {
                'description': '',
                'keywords': [],
                'title': '',
                'error': 'no_url',
            }
            continue

        print(f"  Scraping offer {offer.id} ({i+1}/{total}): {offer.preview_url[:60]}...")
        result = scrape_offer_page(offer.preview_url)

        if result:
            cache[offer_id_str] = {
                'title': result.get('title', ''),
                'description': result.get('description', ''),
                'keywords': result.get('keywords', []),
                'headings': result.get('headings', []),
            }
        else:
            cache[offer_id_str] = {
                'description': '',
                'keywords': [],
                'title': '',
                'error': 'scrape_failed',
            }

        new_scrapes += 1
        if new_scrapes % 10 == 0:
            # Save periodically
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(cache, f, indent=2)

        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Final save
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"  Scraping complete: {new_scrapes} new, {len(cache)} total cached")
    return cache


if __name__ == '__main__':
    catalog = OfferCatalog('data/offers.csv')
    scrape_all_offers(catalog)
