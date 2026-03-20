"""Parse and index the offers catalog with geo-targeting extraction."""

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Offer:
    id: int
    name: str
    clean_name: str
    preview_url: str
    payout_type: str
    payout_amount: float
    payout_percentage: float
    target_countries: list[str]
    restrictions: list[str]
    scraped_description: str = ""
    scraped_keywords: list[str] = field(default_factory=list)


# 2-letter country codes (ISO 3166-1 alpha-2 subset we expect)
COUNTRY_CODE_RE = re.compile(r'^[A-Z]{2}$')
ALL_COUNTRIES_RE = re.compile(r'^All\s+Cou[nt]tries$', re.IGNORECASE)
RESTRICTION_RE = re.compile(r'^\(.*\)$')

# Normalize non-standard codes to ISO
COUNTRY_NORMALIZE = {
    'UK': 'GB',
}


def parse_payout(amount_str: str) -> float:
    """Parse payout amount like '$11.25' to float."""
    cleaned = amount_str.strip().replace('$', '').replace(',', '')
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_offer_name(name: str) -> tuple[str, list[str], list[str]]:
    """
    Parse offer name to extract clean name, target countries, and restrictions.

    Returns (clean_name, target_countries, restrictions).
    """
    parts = [p.strip() for p in name.split(' - ')]

    countries = []
    restrictions = []
    name_parts = []
    found_geo = False

    # Scan from right to find geo and restrictions
    for part in reversed(parts):
        if not found_geo:
            # Check if this is a restriction like (Proof Needed)
            if RESTRICTION_RE.match(part):
                restrictions.append(part.strip('()'))
                continue

            # Check for "All Countries" / "All Coutries"
            if ALL_COUNTRIES_RE.match(part):
                countries = ['ALL']
                found_geo = True
                continue

            # Check for country codes like "US" or "US,CA"
            tokens = [t.strip() for t in part.split(',')]
            if all(COUNTRY_CODE_RE.match(t) for t in tokens) and len(tokens) >= 1:
                for t in tokens:
                    normalized = COUNTRY_NORMALIZE.get(t, t)
                    countries.append(normalized)
                found_geo = True
                continue

            # Check for patterns like "iOS Only", "Android Only" embedded without parens
            if part.lower() in ('ios only', 'android only'):
                restrictions.append(part)
                continue

        # Everything else is part of the name
        name_parts.append(part)

    clean_name = ' - '.join(reversed(name_parts))

    # If no geo found, default to ALL
    if not countries:
        countries = ['ALL']

    return clean_name, countries, restrictions


class OfferCatalog:
    def __init__(self, csv_path: str | Path):
        self.offers: dict[int, Offer] = {}
        self._by_country: dict[str, list[Offer]] = {}
        self._load(csv_path)

    def _load(self, csv_path: str | Path):
        with open(csv_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                offer_id = int(row['Offer ID'])
                name = row['Name'].strip()
                clean_name, countries, restrictions = parse_offer_name(name)

                offer = Offer(
                    id=offer_id,
                    name=name,
                    clean_name=clean_name,
                    preview_url=row['Preview URL'].strip(),
                    payout_type=row['Payout Type'].strip(),
                    payout_amount=parse_payout(row['Payout Amount']),
                    payout_percentage=parse_payout(row['Payout Percentage'].replace('%', '')),
                    target_countries=countries,
                    restrictions=restrictions,
                )
                self.offers[offer_id] = offer

                for country in countries:
                    self._by_country.setdefault(country, []).append(offer)

    def load_scraped_data(self, json_path: str | Path):
        """Load cached scraped descriptions/keywords into offers."""
        path = Path(json_path)
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        for offer_id_str, info in data.items():
            offer_id = int(offer_id_str)
            if offer_id in self.offers:
                self.offers[offer_id].scraped_description = info.get('description', '')
                self.offers[offer_id].scraped_keywords = info.get('keywords', [])

    def get_eligible_offers(self, country_code: str | None) -> list[Offer]:
        """Get offers eligible for a given country code, sorted by payout desc."""
        eligible = []

        # Always include ALL-targeted offers
        eligible.extend(self._by_country.get('ALL', []))

        # Add country-specific offers if we know the country
        if country_code:
            eligible.extend(self._by_country.get(country_code.upper(), []))

        # Deduplicate (an offer could appear in both ALL and country)
        seen = set()
        unique = []
        for offer in eligible:
            if offer.id not in seen:
                seen.add(offer.id)
                unique.append(offer)

        # Sort by payout amount descending
        unique.sort(key=lambda o: o.payout_amount, reverse=True)
        return unique

    def get_all_offers(self) -> list[Offer]:
        return list(self.offers.values())


if __name__ == '__main__':
    catalog = OfferCatalog('data/offers.csv')
    print(f"Loaded {len(catalog.offers)} offers")

    # Test parsing
    test_cases = [
        "Yourself First - US,CA - (Proof Needed)",
        "Home Services - Redostar - Windows - (Proof Needed) - US",
        "SisalFunClub - CPR - iOS Only - IT - (Proof Needed)",
        "Titanium Cutting Board - KatuChef - CTC $59.00 - All Coutries",
        "EKTA World Wide Travel Insurance + Covid - All Countries",
    ]
    for name in test_cases:
        clean, countries, restrictions = parse_offer_name(name)
        print(f"  {name!r}")
        print(f"    -> name={clean!r}, countries={countries}, restrictions={restrictions}")

    # Test eligibility
    for code in ['US', 'GB', 'EG', None]:
        eligible = catalog.get_eligible_offers(code)
        print(f"\n  Eligible for {code}: {len(eligible)} offers")
        if eligible:
            print(f"    Top 3: {[(o.clean_name[:40], o.payout_amount) for o in eligible[:3]]}")
