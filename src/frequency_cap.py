"""Track impressions/clicks per offer for frequency capping."""

from collections import defaultdict


class FrequencyCap:
    def __init__(self, max_total: int = 50, max_daily: int = 10):
        self.max_total = max_total
        self.max_daily = max_daily
        self._total: dict[int, int] = defaultdict(int)
        self._daily: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def can_serve(self, offer_id: int, date: str) -> bool:
        """Check if this offer can still be served."""
        if self._total[offer_id] >= self.max_total:
            return False
        if self._daily[offer_id][date] >= self.max_daily:
            return False
        return True

    def record_serve(self, offer_id: int, date: str):
        """Record a serve for frequency tracking."""
        self._total[offer_id] += 1
        self._daily[offer_id][date] += 1

    def get_stats(self) -> dict:
        return {
            'offers_with_clicks': len(self._total),
            'total_clicks_tracked': sum(self._total.values()),
        }
