"""Generate comparison report between baseline and engine results."""

from src.baseline import BaselineData
from src.engine import EngineResults


def generate_report(
    baseline: BaselineData,
    runs: dict[str, EngineResults],
    output_path: str = 'output/report.md',
) -> str:
    """Generate a markdown comparison report."""
    lines = []
    lines.append("# Ad Delivery Engine — Performance Report\n")

    # Build table header
    run_names = list(runs.keys())
    header = "| Metric | Baseline |"
    separator = "|--------|----------|"
    for name in run_names:
        header += f" {name} |"
        separator += "----------|"
    lines.append(header)
    lines.append(separator)

    # Metrics
    def row(metric: str, baseline_val: str, engine_vals: list[str]) -> str:
        r = f"| {metric} | {baseline_val} |"
        for v in engine_vals:
            r += f" {v} |"
        return r

    lines.append(row(
        "Opportunities",
        str(baseline.num_opportunities),
        [str(r.num_opportunities) for r in runs.values()],
    ))
    lines.append(row(
        "Impressions",
        str(baseline.num_impressions),
        [str(r.num_impressions) for r in runs.values()],
    ))
    lines.append(row(
        "Fill Rate",
        f"{baseline.fill_rate:.1%}",
        [f"{r.fill_rate:.1%}" for r in runs.values()],
    ))
    lines.append(row(
        "Unique Offers Served",
        "—",
        [str(r.unique_offers) for r in runs.values()],
    ))
    lines.append(row(
        "Avg Relevance Score",
        "—",
        [f"{r.avg_relevance_score:.4f}" for r in runs.values()],
    ))
    lines.append(row(
        "Geo Match Rate",
        "—",
        [f"{r.geo_match_rate:.1%}" for r in runs.values()],
    ))

    # Payout type breakdown
    lines.append("\n## Payout Type Breakdown\n")
    for name, result in runs.items():
        breakdown = result.payout_type_breakdown()
        lines.append(f"**{name}**:")
        for ptype, count in sorted(breakdown.items()):
            pct = count / result.num_impressions * 100 if result.num_impressions else 0
            lines.append(f"- {ptype}: {count} ({pct:.1f}%)")
        lines.append("")

    # Top served offers
    lines.append("## Top 10 Served Offers\n")
    for name, result in runs.items():
        lines.append(f"**{name}**:\n")
        offer_counts: dict[int, tuple[str, int]] = {}
        for e in result.impressions:
            if e.offer_id:
                if e.offer_id not in offer_counts:
                    offer_counts[e.offer_id] = (e.offer_name or '', 0)
                n, c = offer_counts[e.offer_id]
                offer_counts[e.offer_id] = (n, c + 1)

        sorted_offers = sorted(offer_counts.items(), key=lambda x: x[1][1], reverse=True)[:10]
        lines.append("| Offer ID | Name | Impressions |")
        lines.append("|----------|------|-------------|")
        for oid, (oname, count) in sorted_offers:
            lines.append(f"| {oid} | {oname[:40]} | {count} |")
        lines.append("")

    # Selection method breakdown (only for hybrid runs)
    hybrid_runs = {name: result for name, result in runs.items()
                   if any(e.selection_method != 'keyword' for e in result.impressions)}
    if hybrid_runs:
        lines.append("## Selection Method Breakdown\n")
        for name, result in hybrid_runs.items():
            breakdown = result.selection_method_breakdown()
            total = sum(breakdown.values())
            lines.append(f"**{name}**:")
            for method, count in sorted(breakdown.items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total else 0
                lines.append(f"- {method}: {count} ({pct:.1f}%)")
            lines.append("")

    # No-fill reasons
    lines.append("## No-Fill Reasons\n")
    for name, result in runs.items():
        reasons: dict[str, int] = {}
        for e in result.events:
            if not e.filled:
                r = e.no_fill_reason or 'unknown'
                reasons[r] = reasons.get(r, 0) + 1
        lines.append(f"**{name}**:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(f"- {reason}: {count}")
        lines.append("")

    report = "\n".join(lines)

    # Write to file
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    return report
