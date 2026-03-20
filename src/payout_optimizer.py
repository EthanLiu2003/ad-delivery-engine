"""Normalized payout scoring for offer ranking."""

import math

from src.offer_catalog import Offer, OfferCatalog


class PayoutScorer:
    """Pre-computes normalized payout scores (0-1) for all offers using log normalization."""

    def __init__(self, catalog: OfferCatalog):
        self._scores: dict[int, float] = {}
        self._compute_all(catalog)

    def _compute_all(self, catalog: OfferCatalog):
        for offer in catalog.get_all_offers():
            self._scores[offer.id] = self._score(offer)

    @staticmethod
    def _score(offer: Offer) -> float:
        if offer.payout_amount > 0:
            return min(math.log(1 + offer.payout_amount) / math.log(1 + 100), 1.0)
        if offer.payout_percentage > 0:
            return 0.3
        return 0.0

    def get_payout_score(self, offer_id: int) -> float:
        return self._scores.get(offer_id, 0.0)

    def rank_by_payout(self, offers: list[Offer]) -> list[Offer]:
        return sorted(offers, key=lambda o: self._scores.get(o.id, 0), reverse=True)


if __name__ == '__main__':
    catalog = OfferCatalog('data/offers.csv')
    catalog.load_scraped_data('data/offer_descriptions.json')
    scorer = PayoutScorer(catalog)

    all_offers = catalog.get_all_offers()
    ranked = scorer.rank_by_payout(all_offers)

    print("Top 20 offers by payout score:")
    for o in ranked[:20]:
        score = scorer.get_payout_score(o.id)
        print(f"  #{o.id} score={score:.4f} {o.payout_type} "
              f"${o.payout_amount}/{o.payout_percentage}% — {o.clean_name[:50]}")
