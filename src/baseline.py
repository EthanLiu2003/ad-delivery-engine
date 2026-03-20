"""Load baseline data for comparison."""

import csv
from pathlib import Path


class BaselineData:
    def __init__(self, data_dir: str | Path = 'data'):
        data_dir = Path(data_dir)
        self.opportunities = self._load_csv(data_dir / 'ad-opportunities-3303.csv')
        self.impressions = self._load_csv(data_dir / 'ad-impressions-3303.csv')
        self.clicks = self._load_csv(data_dir / 'ad-clicks-3303.csv')

    def _load_csv(self, path: Path) -> list[dict]:
        rows = []
        with open(path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    @property
    def num_opportunities(self) -> int:
        return len(self.opportunities)

    @property
    def num_impressions(self) -> int:
        return len(self.impressions)

    @property
    def num_clicks(self) -> int:
        return len(self.clicks)

    @property
    def ctr(self) -> float:
        if self.num_impressions == 0:
            return 0.0
        return self.num_clicks / self.num_impressions

    @property
    def fill_rate(self) -> float:
        if self.num_opportunities == 0:
            return 0.0
        return self.num_impressions / self.num_opportunities

    def summary(self) -> dict:
        return {
            'opportunities': self.num_opportunities,
            'impressions': self.num_impressions,
            'clicks': self.num_clicks,
            'fill_rate': f"{self.fill_rate:.1%}",
            'ctr': f"{self.ctr:.1%}",
        }


if __name__ == '__main__':
    baseline = BaselineData()
    for k, v in baseline.summary().items():
        print(f"  {k}: {v}")
