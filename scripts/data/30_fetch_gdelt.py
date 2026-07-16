"""Estima costo y, con confirmacion explicita, consulta GDELT en BigQuery."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import sha256_file, utc_now, write_json


SQL_PATH = Path(__file__).resolve().parent / "sql" / "gdelt_daily_tone_candidates.sql"
ALLOWED_SCOPES = {"exact_ndx", "nasdaq_market", "broad_nasdaq"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--billing-project", required=True)
    parser.add_argument("--start", default="2015-02-19")
    parser.add_argument("--end", default="2025-04-22")
    parser.add_argument(
        "--maximum-gib",
        type=float,
        default=850.0,
        help="tope duro de bytes facturables para la consulta real",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="ejecuta la consulta; sin esta bandera solo hace dry-run",
    )
    args = parser.parse_args()

    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise SystemExit(
            "Instale requirements-data.txt antes de consultar BigQuery"
        ) from exc

    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date()
    if start >= end:
        raise ValueError("--start debe ser anterior a --end")
    if args.maximum_gib <= 0:
        raise ValueError("--maximum-gib debe ser positivo")

    sql = SQL_PATH.read_text(encoding="utf-8")
    params = [
        bigquery.ScalarQueryParameter("start_date", "DATE", start),
        bigquery.ScalarQueryParameter("end_date", "DATE", end),
    ]
    client = bigquery.Client(project=args.billing_project)
    dry_config = bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False, query_parameters=params
    )
    dry_job = client.query(sql, job_config=dry_config)
    estimated = int(dry_job.total_bytes_processed or 0)
    gib = estimated / (1024 ** 3)
    maximum_bytes = int(args.maximum_gib * (1024 ** 3))
    print(f"Dry-run OK: {estimated} bytes ({gib:.2f} GiB) estimados")
    if estimated > maximum_bytes:
        raise RuntimeError(
            f"La estimacion supera el tope de {args.maximum_gib:.2f} GiB; "
            "no se ejecutara la consulta"
        )
    if not args.execute:
        print("No se ejecuto la consulta. Revise el costo y repita con --execute.")
        return

    job_config = bigquery.QueryJobConfig(
        use_query_cache=False,
        query_parameters=params,
        maximum_bytes_billed=maximum_bytes,
    )
    job = client.query(sql, job_config=job_config)
    frame = job.result().to_dataframe()
    if frame.empty:
        raise ValueError("GDELT no devolvio observaciones")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if not set(frame["scope"]).issubset(ALLOWED_SCOPES):
        raise ValueError("GDELT devolvio un scope inesperado")
    if (frame["n_articles"] <= 0).any() or frame["mean_tone"].isna().any():
        raise ValueError("GDELT devolvio conteos o tonos invalidos")

    output = ROOT / "data" / "source" / "gdelt_daily_tone_candidates.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, date_format="%Y-%m-%d")
    manifest = {
        "manifest_version": 1,
        "created_at_utc": utc_now(),
        "provider": "The GDELT Project via Google BigQuery public dataset",
        "table": "gdelt-bq.gdeltv2.gkg_partitioned",
        "query_file": str(SQL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "query_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
        "parameters": {"start": args.start, "end": args.end},
        "billing_project": args.billing_project,
        "job_id": job.job_id,
        "estimated_bytes_processed": estimated,
        "maximum_bytes_billed": maximum_bytes,
        "actual_bytes_processed": int(job.total_bytes_processed or 0),
        "output_file": "data/source/gdelt_daily_tone_candidates.csv",
        "output_sha256": sha256_file(output),
        "rows": int(len(frame)),
        "scope_counts": {
            str(key): int(value) for key, value in frame.groupby("scope").size().items()
        },
        "selection_status": (
            "candidate_scopes_only; final scope must be selected from coverage "
            "and relevance audit before model execution"
        ),
    }
    manifest_path = ROOT / "data" / "manifests" / "gdelt_source.json"
    write_json(manifest_path, manifest)
    print(f"OK: {output.relative_to(ROOT)} ({len(frame)} filas)")
    print(f"Manifiesto: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
