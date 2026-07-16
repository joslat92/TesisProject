"""Audita cobertura de los scopes GDELT sin usar resultados predictivos."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import sha256_file, utc_now, write_json
from gdelt_utils import aggregate_to_market_dates, longest_missing_run


def main() -> None:
    config = yaml.safe_load((ROOT / "config_data.yaml").read_text(encoding="utf-8"))
    candidates_path = ROOT / "data" / "source" / "gdelt_daily_tone_candidates.csv"
    if not candidates_path.exists():
        raise FileNotFoundError("Ejecute primero scripts/data/30_fetch_gdelt.py --execute")

    ndx = pd.read_csv(ROOT / "data" / "source" / "nasdaq100_fred.csv", parse_dates=["Date"])
    vix = pd.read_csv(ROOT / "data" / "source" / "vix_cboe.csv", parse_dates=["Date"])
    candidates = pd.read_csv(candidates_path, parse_dates=["Date"])
    market_dates = pd.Series(sorted(set(ndx["Date"]) & set(vix["Date"])))
    market_dates = market_dates[
        market_dates.between(config["period"]["start"], config["period"]["end"])
    ].reset_index(drop=True)

    audits = {}
    samples = []
    for scope in config["gdelt"]["allowed_scopes"]:
        aggregated = aggregate_to_market_dates(candidates, market_dates, scope)
        aligned = pd.DataFrame({"Date": market_dates}).merge(
            aggregated, on="Date", how="left"
        )
        present = aligned["Sentiment_GDELT"].notna()
        article_counts = aligned.loc[present, "GDELT_Articles"]
        audits[scope] = {
            "market_dates": int(len(aligned)),
            "covered_market_dates": int(present.sum()),
            "coverage_ratio": float(present.mean()),
            "longest_missing_market_date_run": longest_missing_run(present),
            "missing_dates_first_20": aligned.loc[~present, "Date"]
                .dt.date.astype(str).head(20).tolist(),
            "articles_total": int(article_counts.sum()),
            "articles_per_covered_date": {
                "min": int(article_counts.min()),
                "median": float(article_counts.median()),
                "p95": float(article_counts.quantile(0.95)),
                "max": int(article_counts.max()),
            },
        }
        scope_rows = candidates.loc[candidates["scope"] == scope].sort_values("Date")
        if "example_urls" in scope_rows:
            sample_count = min(30, len(scope_rows))
            indices = [round(i) for i in pd.Series(
                range(len(scope_rows)), dtype=float
            ).quantile([i / max(sample_count - 1, 1) for i in range(sample_count)])]
            sampled = scope_rows.iloc[sorted(set(indices))][
                ["Date", "scope", "n_articles", "example_urls"]
            ]
            samples.append(sampled)

    eligible = [
        scope for scope, audit in audits.items()
        if audit["coverage_ratio"] >= 0.98
        and audit["longest_missing_market_date_run"] <= 2
    ]
    coverage_only_candidate = (
        "strict_ndx" if "strict_ndx" in eligible
        else "broad_nasdaq" if "broad_nasdaq" in eligible
        else None
    )
    report = {
        "created_at_utc": utc_now(),
        "candidate_file_sha256": sha256_file(candidates_path),
        "selection_guard": (
            "This audit does not inspect model performance. Scope selection also "
            "requires a documented manual relevance review of sampled URLs."
        ),
        "coverage_acceptance_thresholds": {
            "minimum_ratio": 0.98,
            "maximum_missing_run": 2,
        },
        "coverage_only_candidate": coverage_only_candidate,
        "scope_audits": audits,
        "manual_relevance_review_status": "pending",
    }
    output = ROOT / "data" / "quality" / "gdelt_candidate_audit.json"
    write_json(output, report)
    if samples:
        sample_path = ROOT / "data" / "quality" / "gdelt_relevance_sample.csv"
        pd.concat(samples, ignore_index=True).to_csv(
            sample_path, index=False, date_format="%Y-%m-%d"
        )
    print(f"OK: {output.relative_to(ROOT)}")
    for scope, audit in audits.items():
        print(
            f"{scope}: coverage={audit['coverage_ratio']:.4f}, "
            f"max_gap={audit['longest_missing_market_date_run']}"
        )
    print(f"Candidato solo por cobertura: {coverage_only_candidate}")


if __name__ == "__main__":
    main()
