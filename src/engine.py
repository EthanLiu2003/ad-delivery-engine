"""Main ad delivery engine orchestrator."""

import csv
from dataclasses import dataclass, field
from pathlib import Path

from src.chat_stream import ChatStream, Message
from src.frequency_cap import FrequencyCap
from src.geo_resolver import GeoResolver
from src.offer_catalog import Offer, OfferCatalog
from src.offer_scraper import scrape_all_offers
from src.payout_optimizer import PayoutScorer
from src.targeting import KeywordSelector, ScoringWeights
from src.llm_selector import LLMSelector


HYBRID_CONFIDENCE_HIGH = 0.5   # keyword is confident, trust it
HYBRID_CONFIDENCE_LOW = 0.1    # keyword has some signal, LLM validates
# below LOW: keyword is blind, LLM picks from scratch


@dataclass
class AdEvent:
    """A single ad opportunity and its outcome."""
    chat_id: str
    visitor_id: str
    session_id: str
    message_id: str
    country_code: str
    filled: bool
    offer_id: int | None = None
    offer_name: str | None = None
    offer_payout_type: str | None = None
    offer_payout_amount: float = 0.0
    relevance_score: float = 0.0
    created_at: str = ''
    no_fill_reason: str = ''
    selection_method: str = 'keyword'


@dataclass
class EngineResults:
    events: list[AdEvent] = field(default_factory=list)

    @property
    def opportunities(self) -> list[AdEvent]:
        return self.events

    @property
    def impressions(self) -> list[AdEvent]:
        return [e for e in self.events if e.filled]

    @property
    def num_opportunities(self) -> int:
        return len(self.events)

    @property
    def num_impressions(self) -> int:
        return len(self.impressions)

    @property
    def fill_rate(self) -> float:
        if not self.events:
            return 0.0
        return self.num_impressions / self.num_opportunities

    @property
    def avg_relevance_score(self) -> float:
        imps = self.impressions
        if not imps:
            return 0.0
        return sum(e.relevance_score for e in imps) / len(imps)

    @property
    def unique_offers(self) -> int:
        return len({e.offer_id for e in self.impressions if e.offer_id})

    @property
    def geo_match_rate(self) -> float:
        with_geo = [e for e in self.events if e.country_code]
        if not with_geo:
            return 0.0
        return len(with_geo) / len(self.events)

    def payout_type_breakdown(self) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for e in self.impressions:
            if e.offer_payout_type:
                breakdown[e.offer_payout_type] = breakdown.get(e.offer_payout_type, 0) + 1
        return breakdown

    def selection_method_breakdown(self) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for e in self.impressions:
            breakdown[e.selection_method] = breakdown.get(e.selection_method, 0) + 1
        return breakdown


class AdEngine:
    def __init__(
        self,
        ad_load: float = 0.5,
        mode: str = 'keyword',
        data_dir: str = 'data',
        scrape: bool = False,
    ):
        self.ad_load = ad_load
        self.mode = mode
        self.data_dir = Path(data_dir)

        print(f"Initializing engine: mode={mode}, ad_load={ad_load}")

        # Load data
        print("  Loading offer catalog...")
        self.catalog = OfferCatalog(self.data_dir / 'offers.csv')

        # Scrape if requested
        if scrape:
            print("  Scraping offer preview URLs...")
            scrape_all_offers(self.catalog, self.data_dir / 'offer_descriptions.json')

        # Load scraped data
        self.catalog.load_scraped_data(self.data_dir / 'offer_descriptions.json')
        print(f"  Loaded {len(self.catalog.offers)} offers")

        # Payout scorer
        print("  Computing payout scores...")
        self.scorer = PayoutScorer(self.catalog)

        # Chat stream
        print("  Loading chat data...")
        self.chat_stream = ChatStream(self.data_dir / 'chat-data-3303.csv')
        print(f"  Loaded {len(self.chat_stream.chats)} chats")

        # Geo resolver
        print("  Loading geo data...")
        self.geo_resolver = GeoResolver(self.data_dir / 'ad-opportunities-3303.csv')
        print(f"  Geo: {self.geo_resolver.known_sessions} sessions, "
              f"{self.geo_resolver.known_visitors} visitors")

        # Selector
        all_offers = self.catalog.get_all_offers()
        if mode == 'llm':
            print("  Initializing LLM selector...")
            self.selector = LLMSelector(self.scorer, all_offers)
        elif mode == 'hybrid':
            print("  Initializing hybrid selector (keyword + LLM fallback)...")
            self.keyword_selector = KeywordSelector(self.scorer, offers=all_offers)
            self.llm_selector = LLMSelector(self.scorer, all_offers)
        else:
            self.selector = KeywordSelector(self.scorer, offers=all_offers)

        # Frequency cap
        self.freq_cap = FrequencyCap()

    def run(self) -> EngineResults:
        """Process all chats and return results.

        ad_load is applied as a deterministic per-chat percentage: for each chat,
        we score all opportunities then fill only the top ad_load fraction.
        """
        results = EngineResults()
        chats = self.chat_stream.iter_chats()
        total_chats = len(chats)

        for chat_idx, chat in enumerate(chats):
            if (chat_idx + 1) % 20 == 0 or chat_idx == 0:
                print(f"  Processing chat {chat_idx + 1}/{total_chats}...")

            # Pass 1: score all opportunities in this chat
            scored_events: list[tuple[AdEvent, Offer | None, float]] = []
            messages_so_far: list[Message] = []
            shown_in_chat: dict[int, int] = {}

            for msg in chat.messages:
                messages_so_far.append(msg)

                if msg.role != 'assistant':
                    continue

                geo = self.geo_resolver.get_geo(msg.visitor_id, msg.session_id)
                country_code = geo.country_code if geo else ''

                event = AdEvent(
                    chat_id=chat.chat_id,
                    visitor_id=msg.visitor_id,
                    session_id=msg.session_id,
                    message_id=msg.id,
                    country_code=country_code,
                    filled=False,
                    created_at=msg.created_at,
                )

                eligible = self.catalog.get_eligible_offers(country_code if country_code else None)
                if not eligible:
                    event.no_fill_reason = 'no_eligible_offers'
                    results.events.append(event)
                    continue

                user_msgs = [m for m in messages_so_far if m.role == 'user']
                if self.mode == 'llm':
                    offer, relevance_score = self.selector.select(
                        user_msgs, eligible,
                        all_messages=messages_so_far,
                        chat_id=chat.chat_id,
                    )
                elif self.mode == 'hybrid':
                    offer, relevance_score, best_ctx = self.keyword_selector.select(
                        user_msgs, eligible,
                        all_messages=messages_so_far,
                        country_code=country_code,
                        shown_in_chat=shown_in_chat,
                    )
                    if best_ctx >= HYBRID_CONFIDENCE_HIGH:
                        event.selection_method = 'keyword'
                    elif best_ctx >= HYBRID_CONFIDENCE_LOW:
                        llm_offer, llm_score = self.llm_selector.select(
                            user_msgs, eligible,
                            all_messages=messages_so_far,
                            chat_id=chat.chat_id,
                        )
                        if llm_offer and llm_score > relevance_score:
                            offer = llm_offer
                            relevance_score = llm_score
                            event.selection_method = 'llm_validated'
                        else:
                            event.selection_method = 'keyword_confirmed'
                    else:
                        llm_offer, llm_score = self.llm_selector.select(
                            user_msgs, eligible,
                            all_messages=messages_so_far,
                            chat_id=chat.chat_id,
                        )
                        if llm_offer:
                            offer = llm_offer
                            relevance_score = llm_score
                            event.selection_method = 'llm'
                        else:
                            event.selection_method = 'keyword_fallback'
                else:
                    offer, relevance_score, _ = self.selector.select(
                        user_msgs, eligible,
                        all_messages=messages_so_far,
                        country_code=country_code,
                        shown_in_chat=shown_in_chat,
                    )

                if offer:
                    shown_in_chat[offer.id] = shown_in_chat.get(offer.id, 0) + 1

                scored_events.append((event, offer, relevance_score))

            # Pass 2: apply ad_load — fill top N by score, where N = ad_load * total
            num_opportunities = len(scored_events)
            num_to_fill = round(self.ad_load * num_opportunities)

            # Rank by relevance score descending, pick top N
            ranked = sorted(
                range(num_opportunities),
                key=lambda i: scored_events[i][2],
                reverse=True,
            )
            fill_indices = set(ranked[:num_to_fill])

            for i, (event, offer, relevance_score) in enumerate(scored_events):
                if i not in fill_indices:
                    event.no_fill_reason = 'ad_load_gate'
                    results.events.append(event)
                    continue

                date = event.created_at[:10] if event.created_at else ''
                if offer and self.freq_cap.can_serve(offer.id, date):
                    event.filled = True
                    event.offer_id = offer.id
                    event.offer_name = offer.clean_name
                    event.offer_payout_type = offer.payout_type
                    event.offer_payout_amount = offer.payout_amount
                    event.relevance_score = relevance_score
                    self.freq_cap.record_serve(offer.id, date)
                else:
                    event.no_fill_reason = 'no_suitable_offer' if not offer else 'freq_cap'

                results.events.append(event)

        if self.mode == 'llm' and hasattr(self.selector, 'api_calls'):
            print(f"  LLM API calls made: {self.selector.api_calls}")
        elif self.mode == 'hybrid':
            print(f"  LLM API calls made: {self.llm_selector.api_calls}")
            breakdown = results.selection_method_breakdown()
            print(f"  Selection method breakdown: {breakdown}")

        return results

    def export_results(self, results: EngineResults, output_dir: str = 'output'):
        """Export results to CSV files mirroring baseline format."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        prefix = f"{self.mode}_load{int(self.ad_load * 100)}"

        # Opportunities
        with open(out / f'{prefix}_opportunities.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'message_id', 'chat_id', 'visitor_id', 'session_id',
                'country_code', 'filled', 'offer_id', 'created_at',
            ])
            writer.writeheader()
            for e in results.events:
                writer.writerow({
                    'message_id': e.message_id,
                    'chat_id': e.chat_id,
                    'visitor_id': e.visitor_id,
                    'session_id': e.session_id,
                    'country_code': e.country_code,
                    'filled': e.filled,
                    'offer_id': e.offer_id or '',
                    'created_at': e.created_at,
                })

        # Impressions
        with open(out / f'{prefix}_impressions.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'message_id', 'chat_id', 'visitor_id', 'session_id',
                'country_code', 'offer_id', 'offer_name', 'payout_type',
                'payout_amount', 'relevance_score', 'selection_method', 'created_at',
            ])
            writer.writeheader()
            for e in results.impressions:
                writer.writerow({
                    'message_id': e.message_id,
                    'chat_id': e.chat_id,
                    'visitor_id': e.visitor_id,
                    'session_id': e.session_id,
                    'country_code': e.country_code,
                    'offer_id': e.offer_id,
                    'offer_name': e.offer_name,
                    'payout_type': e.offer_payout_type,
                    'payout_amount': e.offer_payout_amount,
                    'relevance_score': round(e.relevance_score, 4),
                    'selection_method': e.selection_method,
                    'created_at': e.created_at,
                })

        print(f"  Exported to {out}/{prefix}_*.csv")

    def export_baseline_comparison(
        self,
        results: EngineResults,
        baseline_opps_path: str = 'data/ad-opportunities-3303.csv',
        baseline_impressions_path: str = 'data/ad-impressions-3303.csv',
        baseline_clicks_path: str = 'data/ad-clicks-3303.csv',
        output_dir: str = 'output',
    ):
        """Compare engine results against baseline opportunities/impressions/clicks.

        Matches baseline opportunities to engine events by session_id + visitor_id
        + closest timestamp, and outputs a side-by-side CSV.
        """
        from datetime import datetime

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.mode}_load{int(self.ad_load * 100)}"

        def parse_ts(s: str) -> datetime | None:
            try:
                return datetime.strptime(s.strip(), '%Y-%m-%d %H:%M:%S')
            except (ValueError, AttributeError):
                return None

        # Load baseline CSVs
        def load_csv(path: str) -> list[dict]:
            with open(path, encoding='utf-8-sig') as f:
                return list(csv.DictReader(f))

        baseline_opps = load_csv(baseline_opps_path)
        baseline_imps = load_csv(baseline_impressions_path)
        baseline_clicks = load_csv(baseline_clicks_path)

        # Index impressions and clicks by opportunity id
        imp_by_opp = {r['ad_opportunity_id']: r for r in baseline_imps}
        click_by_opp = {r['ad_opportunity_id']: r for r in baseline_clicks}

        # Index engine events by (session_id, visitor_id)
        engine_by_session: dict[tuple[str, str], list[AdEvent]] = {}
        for e in results.events:
            key = (e.session_id, e.visitor_id)
            engine_by_session.setdefault(key, []).append(e)

        # Match each baseline opportunity to the closest engine event
        comparison_rows = []
        matched = 0
        unmatched_sessions = 0

        for opp in baseline_opps:
            opp_id = opp['id']
            sid = opp['session_id']
            vid = opp['visitor_id']
            opp_ts = parse_ts(opp['created_at'])
            opp_country = opp.get('geo_country_code', '')

            had_impression = opp_id in imp_by_opp
            had_click = opp_id in click_by_opp
            click_url = click_by_opp[opp_id].get('ad_url', '') if had_click else ''

            row = {
                'baseline_opp_id': opp_id,
                'session_id': sid,
                'visitor_id': vid,
                'baseline_country': opp_country,
                'baseline_timestamp': opp.get('created_at', ''),
                'baseline_had_impression': had_impression,
                'baseline_had_click': had_click,
                'baseline_click_url': click_url,
                'engine_matched': False,
                'engine_message_id': '',
                'engine_filled': '',
                'engine_offer_id': '',
                'engine_offer_name': '',
                'engine_payout_type': '',
                'engine_payout_amount': '',
                'engine_relevance_score': '',
                'engine_no_fill_reason': '',
                'engine_timestamp': '',
                'time_diff_seconds': '',
            }

            candidates = engine_by_session.get((sid, vid), [])
            if not candidates or not opp_ts:
                if not candidates:
                    unmatched_sessions += 1
                comparison_rows.append(row)
                continue

            # Find closest engine event by timestamp
            best_event = None
            best_diff = float('inf')
            for e in candidates:
                e_ts = parse_ts(e.created_at)
                if e_ts:
                    diff = abs((e_ts - opp_ts).total_seconds())
                    if diff < best_diff:
                        best_diff = diff
                        best_event = e

            if best_event and best_diff <= 60:
                matched += 1
                row['engine_matched'] = True
                row['engine_message_id'] = best_event.message_id
                row['engine_filled'] = best_event.filled
                row['engine_offer_id'] = best_event.offer_id or ''
                row['engine_offer_name'] = best_event.offer_name or ''
                row['engine_payout_type'] = best_event.offer_payout_type or ''
                row['engine_payout_amount'] = best_event.offer_payout_amount if best_event.filled else ''
                row['engine_relevance_score'] = round(best_event.relevance_score, 4) if best_event.filled else ''
                row['engine_no_fill_reason'] = best_event.no_fill_reason
                row['engine_timestamp'] = best_event.created_at
                row['time_diff_seconds'] = round(best_diff, 1)

            comparison_rows.append(row)

        # Write comparison CSV
        fieldnames = list(comparison_rows[0].keys())
        comp_path = out / f'{prefix}_baseline_comparison.csv'
        with open(comp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)

        # Print summary
        total = len(baseline_opps)
        engine_filled = sum(1 for r in comparison_rows if r['engine_filled'] is True)
        baseline_clicked = sum(1 for r in comparison_rows if r['baseline_had_click'])
        print(f"\n  Baseline Comparison ({comp_path}):")
        print(f"    Baseline opportunities: {total}")
        print(f"    Matched to engine events: {matched}/{total} "
              f"({unmatched_sessions} had no session in chat data)")
        print(f"    Engine filled (of matched): {engine_filled}/{matched}")
        print(f"    Baseline had clicks: {baseline_clicked}/{total}")

        # Show what engine would serve for clicked baseline opportunities
        clicked_rows = [r for r in comparison_rows if r['baseline_had_click'] and r['engine_matched']]
        if clicked_rows:
            print(f"\n  Engine picks for baseline-clicked opportunities ({len(clicked_rows)} matched):")
            for r in clicked_rows:
                fill = "FILLED" if r['engine_filled'] else f"NO-FILL ({r['engine_no_fill_reason']})"
                offer_info = f"#{r['engine_offer_id']} {r['engine_offer_name']}" if r['engine_filled'] else ""
                score = f" score={r['engine_relevance_score']}" if r['engine_relevance_score'] != '' else ""
                print(f"    {r['baseline_opp_id'][:8]}... -> {fill} {offer_info}{score}")
                if r['baseline_click_url']:
                    print(f"      baseline clicked: {r['baseline_click_url'][:80]}")

        return comparison_rows
