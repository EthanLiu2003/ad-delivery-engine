"""Build geo lookup from ad-opportunities data, joined by visitor_id + session_id."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GeoInfo:
    country_code: str
    country_name: str
    region: str
    city: str


class GeoResolver:
    def __init__(self, opportunities_csv: str | Path):
        # Keyed by (visitor_id, session_id)
        self._by_visitor_session: dict[tuple[str, str], GeoInfo] = {}
        # Keyed by visitor_id alone (fallback)
        self._by_visitor: dict[str, GeoInfo] = {}
        self._load(opportunities_csv)

    def _load(self, csv_path: str | Path):
        with open(csv_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                geo = GeoInfo(
                    country_code=row.get('geo_country_code', ''),
                    country_name=row.get('geo_country_name', ''),
                    region=row.get('geo_region', ''),
                    city=row.get('geo_city', ''),
                )

                visitor_id = row.get('visitor_id', '')
                session_id = row.get('session_id', '')

                if visitor_id and session_id:
                    self._by_visitor_session[(visitor_id, session_id)] = geo
                if visitor_id:
                    self._by_visitor[visitor_id] = geo

    def get_geo(self, visitor_id: str, session_id: str) -> GeoInfo | None:
        """Look up geo info, first by (visitor_id, session_id), then by visitor_id alone."""
        geo = self._by_visitor_session.get((visitor_id, session_id))
        if geo:
            return geo
        return self._by_visitor.get(visitor_id)

    @property
    def known_visitors(self) -> int:
        return len(self._by_visitor)

    @property
    def known_sessions(self) -> int:
        return len(self._by_visitor_session)


if __name__ == '__main__':
    resolver = GeoResolver('data/ad-opportunities-3303.csv')
    print(f"Known visitor+session pairs: {resolver.known_sessions}")
    print(f"Known visitors (fallback): {resolver.known_visitors}")

    # Show country distribution
    countries: dict[str, int] = {}
    for geo in resolver._by_visitor_session.values():
        countries[geo.country_code] = countries.get(geo.country_code, 0) + 1
    for code, count in sorted(countries.items(), key=lambda x: -x[1]):
        print(f"  {code}: {count}")
