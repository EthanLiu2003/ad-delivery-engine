"""Multi-signal relevance scoring for ad selection."""

import math
import re
from dataclasses import dataclass

from src.chat_stream import Message
from src.offer_catalog import Offer
from src.payout_optimizer import PayoutScorer

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
    'it', 'its', 'you', 'your', 'we', 'our', 'they', 'their', 'my',
    'not', 'no', 'all', 'each', 'every', 'any', 'some', 'more', 'most',
    'other', 'into', 'up', 'out', 'about', 'than', 'just', 'also',
    'get', 'how', 'what', 'when', 'where', 'who', 'which', 'why',
    'so', 'if', 'then', 'as', 'only', 'very', 'too', 'here', 'there',
    'now', 'new', 'one', 'two', 'first', 'last', 'free',
    'yes', 'yeah', 'okay', 'ok', 'sure', 'hi', 'hello', 'hey',
    'thanks', 'thank', 'please', 'like', 'know', 'think', 'want',
    'need', 'going', 'got', 'really', 'well', 'good', 'right',
    'said', 'say', 'see', 'look', 'make', 'way', 'much', 'many',
    'been', 'being', 'come', 'came', 'still', 'even', 'back',
    'let', 'tell', 'told', 'thing', 'things', 'lot', 'take', 'took',
    'i', 'me', 'him', 'her', 'them', 'us', 'she', 'he',
}


@dataclass
class ScoringWeights:
    w_context: float = 0.70
    w_geo: float = 0.25
    w_repeat: float = 0.05
    w_payout: float = 0.00


def tokenize(text: str) -> list[str]:
    """Extract meaningful tokens from text."""
    words = re.findall(r'[a-z]+', text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def extract_ngrams(tokens: list[str], max_n: int = 3) -> list[tuple[str, ...]]:
    """Extract unigrams, bigrams, and trigrams from a token list."""
    ngrams = []
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
    return ngrams


class TFIDFTable:
    """IDF weights computed from all offer descriptions."""

    def __init__(self, offers: list[Offer]):
        n_docs = len(offers)
        # Document frequency: how many offers contain each n-gram
        df: dict[tuple[str, ...], int] = {}
        for offer in offers:
            offer_tokens = self._offer_tokens(offer)
            # Unique n-grams per document
            unique_ngrams = set(extract_ngrams(offer_tokens))
            for ng in unique_ngrams:
                df[ng] = df.get(ng, 0) + 1

        # Compute IDF
        self._idf: dict[tuple[str, ...], float] = {}
        self._max_idf = math.log(n_docs / 1) if n_docs > 0 else 1.0
        for ng, count in df.items():
            self._idf[ng] = math.log(n_docs / (1 + count))

    @staticmethod
    def _offer_tokens(offer: Offer) -> list[str]:
        text = offer.clean_name + ' ' + offer.scraped_description
        tokens = tokenize(text)
        for kw in offer.scraped_keywords:
            tokens.extend(tokenize(kw))
        return tokens

    def get_idf(self, ngram: tuple[str, ...]) -> float:
        return self._idf.get(ngram, self._max_idf)


def extract_chat_ngrams(messages: list[Message], user_weight: float = 2.0) -> dict[tuple[str, ...], float]:
    """Extract weighted n-grams from chat messages. Recent messages weighted higher."""
    ngram_scores: dict[tuple[str, ...], float] = {}
    total = len(messages)

    for i, msg in enumerate(messages):
        recency = 0.5 + 0.5 * (i / max(total - 1, 1))
        role_weight = user_weight if msg.role == 'user' else 1.0
        weight = recency * role_weight

        tokens = tokenize(msg.text)
        for ng in extract_ngrams(tokens):
            ngram_scores[ng] = ngram_scores.get(ng, 0) + weight

    return ngram_scores


def score_offer(offer: Offer, chat_ngrams: dict[tuple[str, ...], float],
                tfidf: TFIDFTable) -> float:
    """Score an offer against chat n-grams using TF-IDF weighting."""
    offer_tokens = TFIDFTable._offer_tokens(offer)
    offer_ngrams = set(extract_ngrams(offer_tokens))

    if not offer_ngrams or not chat_ngrams:
        return 0.0

    score = 0.0
    for ngram in offer_ngrams:
        if ngram in chat_ngrams:
            chat_weight = chat_ngrams[ngram]
            idf = tfidf.get_idf(ngram)
            length_bonus = len(ngram)  # 1x unigram, 2x bigram, 3x trigram
            score += chat_weight * idf * length_bonus

    return score


def normalize_context_score(raw: float, k: float = 110.0) -> float:
    """Normalize raw TF-IDF n-gram score to 0-1."""
    return min(raw / k, 1.0)


def compute_geo_score(offer: Offer, country_code: str) -> float:
    """Compute geo match score: 1.0 if country matches, 0.5 if ALL or unknown, 0.0 if no match."""
    if not country_code:
        return 0.5
    if country_code.upper() in offer.target_countries:
        return 1.0
    if 'ALL' in offer.target_countries:
        return 0.5
    return 0.0


def compute_repeat_penalty(offer_id: int, shown_in_chat: dict[int, int],
                           diversity_decay: float = 0.5) -> float:
    """Compute repeat freshness with exponential diversity decay.

    Each repeat showing multiplies by e^(-diversity_decay), so offers shown
    multiple times in the same chat get progressively penalized.
    """
    times_shown = shown_in_chat.get(offer_id, 0)
    return math.exp(-diversity_decay * times_shown)


class KeywordSelector:
    def __init__(self, scorer: PayoutScorer, offers: list[Offer] | None = None,
                 weights: ScoringWeights | None = None):
        self.scorer = scorer
        self.weights = weights or ScoringWeights()
        self.tfidf = TFIDFTable(offers) if offers else TFIDFTable([])

    def select(self, user_messages: list[Message], eligible_offers: list[Offer],
               all_messages: list[Message] | None = None,
               country_code: str = '',
               shown_in_chat: dict[int, int] | None = None) -> tuple[Offer | None, float, float]:
        """Select the best offer using multi-signal scoring.

        Returns (offer, final_score, best_context) tuple.
        """
        if not eligible_offers:
            return None, 0.0, 0.0

        if shown_in_chat is None:
            shown_in_chat = {}

        messages_for_keywords = all_messages if all_messages else user_messages
        chat_ngrams = extract_chat_ngrams(messages_for_keywords)

        w = self.weights
        best_offer = None
        best_score = -1.0
        best_context = 0.0

        for offer in eligible_offers:
            context_raw = score_offer(offer, chat_ngrams, self.tfidf)
            context = normalize_context_score(context_raw)
            geo = compute_geo_score(offer, country_code)
            repeat = compute_repeat_penalty(offer.id, shown_in_chat)
            payout = self.scorer.get_payout_score(offer.id)

            final = (w.w_context * context +
                     w.w_geo * geo +
                     w.w_repeat * repeat +
                     w.w_payout * payout)

            if final > best_score:
                best_score = final
                best_offer = offer
                best_context = context

        return best_offer, best_score, best_context
