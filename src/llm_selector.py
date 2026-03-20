"""Claude API-based intelligent ad selection."""

import json
import os

from anthropic import Anthropic

from src.chat_stream import Message
from src.offer_catalog import Offer
from src.payout_optimizer import PayoutScorer
from src.targeting import TFIDFTable, extract_chat_ngrams, score_offer

MODEL = "claude-haiku-4-5-20251001"
MAX_CONTEXT_MESSAGES = 10
MAX_CANDIDATE_OFFERS = 40
RECACHE_AFTER_N_USER_MSGS = 5


class LLMSelector:
    def __init__(self, scorer: PayoutScorer, offers: list[Offer] | None = None):
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required for LLM mode")
        self.client = Anthropic(api_key=api_key)
        self.scorer = scorer
        self.tfidf = TFIDFTable(offers) if offers else TFIDFTable([])
        # Cache: chat_id -> ([(offer_id, score), ...], user_msg_count_at_cache_time)
        self._cache: dict[str, tuple[list[tuple[int, float]], int]] = {}
        self._api_calls = 0

    def select(self, user_messages: list[Message], eligible_offers: list[Offer],
               all_messages: list[Message] | None = None, chat_id: str = '') -> tuple[Offer | None, float]:
        """Select the best offer using Claude API, with per-chat caching.

        Returns (offer, relevance_score) tuple. Score is 0.0-1.0 from LLM.
        """
        if not eligible_offers:
            return None, 0.0

        user_msg_count = len(user_messages)
        offer_map = {o.id: o for o in eligible_offers}

        # Check cache
        if chat_id in self._cache:
            cached_rankings, cached_count = self._cache[chat_id]
            if user_msg_count - cached_count < RECACHE_AFTER_N_USER_MSGS:
                # Use cached ranking
                for oid, score in cached_rankings:
                    if oid in offer_map:
                        return offer_map[oid], score
                # If none of the cached offers are eligible, fall through

        # Build context for LLM
        msgs = all_messages or user_messages
        recent_msgs = msgs[-MAX_CONTEXT_MESSAGES:]
        chat_context = "\n".join(
            f"[{m.role}]: {m.text[:200]}" for m in recent_msgs
        )

        # Build candidate list: all keyword-relevant offers + top by payout
        chat_ngrams = extract_chat_ngrams(recent_msgs)
        scored = [(o, score_offer(o, chat_ngrams, self.tfidf)) for o in eligible_offers]

        # All offers with any keyword relevance
        relevant = [(o, raw) for o, raw in scored if raw > 0]
        relevant.sort(key=lambda x: x[1], reverse=True)
        relevant_ids = {o.id for o, _ in relevant}

        # Backfill with top payout offers up to MAX_CANDIDATE_OFFERS
        by_payout = self.scorer.rank_by_payout(eligible_offers)
        backfill = [o for o in by_payout if o.id not in relevant_ids]

        candidates = [o for o, _ in relevant]
        remaining = MAX_CANDIDATE_OFFERS - len(candidates)
        if remaining > 0:
            candidates.extend(backfill[:remaining])

        offer_lines = []
        for o in candidates:
            desc = o.scraped_description[:100] if o.scraped_description else o.clean_name
            payout_rank = next((i + 1 for i, po in enumerate(by_payout) if po.id == o.id), len(by_payout))
            offer_lines.append(f"ID:{o.id} | payout_rank:{payout_rank} | {o.payout_type} ${o.payout_amount} | {desc}")
        offers_text = "\n".join(offer_lines)

        prompt = f"""You are an ad relevance engine. Given a chat conversation and a list of available offers, select the most relevant offers to show to this user.

Consider:
1. Chat topic and user intent
2. Offer relevance to the conversation
3. Payout score — prefer higher payout score offers when relevance is similar

Chat conversation:
{chat_context}

Available offers:
{offers_text}

Return a JSON array of objects with "id" and "relevance" (1-10 scale, 10 = perfect match) for the top 5 offers, most relevant first.
Example: [{{"id": 1234, "relevance": 8}}, {{"id": 5678, "relevance": 6}}]
Return ONLY the JSON array, no other text."""

        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            self._api_calls += 1

            text = response.content[0].text.strip()
            # Handle possible markdown code blocks
            if '```' in text:
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
            parsed = json.loads(text)

            if isinstance(parsed, list):
                # Support both old format [id, ...] and new format [{"id": ..., "relevance": ...}, ...]
                rankings: list[tuple[int, float]] = []
                for item in parsed:
                    if isinstance(item, dict):
                        oid = int(item['id'])
                        score = min(max(float(item.get('relevance', 5)), 1), 10) / 10.0
                        rankings.append((oid, score))
                    else:
                        rankings.append((int(item), 0.5))

                self._cache[chat_id] = (rankings, user_msg_count)
                for oid, score in rankings:
                    if oid in offer_map:
                        return offer_map[oid], score

        except Exception as e:
            print(f"  LLM API error: {e}")

        # Fallback: return highest payout offer
        if candidates:
            return self.scorer.rank_by_payout(candidates)[0], 0.0
        return None, 0.0

    @property
    def api_calls(self) -> int:
        return self._api_calls
