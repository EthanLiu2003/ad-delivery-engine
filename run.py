"""CLI entry point for the ad delivery engine."""

import argparse
import sys

from src.baseline import BaselineData
from src.engine import AdEngine
from report import generate_report


def main():
    parser = argparse.ArgumentParser(description='Ad Delivery Engine')
    parser.add_argument('--ad-load', type=float, default=0.5,
                        help='Ad load percentage (0.0-1.0)')
    parser.add_argument('--mode', choices=['keyword', 'llm', 'hybrid'], default='keyword',
                        help='Selection mode: keyword (heuristic), llm (Claude API), or hybrid (keyword + LLM fallback)')
    parser.add_argument('--scrape', action='store_true',
                        help='Scrape offer preview URLs (first run only)')
    parser.add_argument('--multi', action='store_true',
                        help='Run at multiple ad loads (0.15, 0.5, 1.0)')
    args = parser.parse_args()

    print("=" * 60)
    print("Ad Delivery Engine")
    print("=" * 60)

    # Load baseline
    print("\nLoading baseline data...")
    baseline = BaselineData()
    print(f"  Baseline: {baseline.num_opportunities} opportunities, "
          f"{baseline.num_impressions} impressions, {baseline.num_clicks} clicks, "
          f"CTR={baseline.ctr:.1%}")

    if args.multi:
        # Run at multiple ad loads
        runs = {}
        for load in [0.15, 0.5, 1.0]:
            run_name = f"{args.mode}_load{int(load * 100)}"
            print(f"\n{'=' * 60}")
            print(f"Running: {run_name}")
            print(f"{'=' * 60}")

            engine = AdEngine(
                ad_load=load,
                mode=args.mode,
                scrape=args.scrape and load == 0.15,  # Only scrape once
            )
            results = engine.run()
            engine.export_results(results)
            engine.export_baseline_comparison(results)
            runs[run_name] = results

            print(f"\n  Results: {results.num_opportunities} opportunities, "
                  f"{results.num_impressions} impressions, "
                  f"avg_relevance={results.avg_relevance_score:.4f}")

        # Generate comparison report
        print(f"\n{'=' * 60}")
        print("Generating report...")
        report = generate_report(baseline, runs)
        print(report)
        print(f"\nReport saved to output/report.md")
    else:
        # Single run
        engine = AdEngine(
            ad_load=args.ad_load,
            mode=args.mode,
            seed=args.seed,
            scrape=args.scrape,
        )
        results = engine.run()
        engine.export_results(results)
        engine.export_baseline_comparison(results)

        run_name = f"{args.mode}_load{int(args.ad_load * 100)}"
        report = generate_report(baseline, {run_name: results})
        print(f"\n{'=' * 60}")
        print(report)
        print(f"\nReport saved to output/report.md")


if __name__ == '__main__':
    main()
