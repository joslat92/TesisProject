"""Utilidades compartidas para la reconstruccion trazable de datos."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


USER_AGENT = "TesisProject-data-reconstruction/1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def download(url: str, destination: Path, attempts: int = 3,
             timeout: int = 90) -> dict:
    """Descarga atomica con reintentos y devuelve metadatos verificables."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
                content_type = response.headers.get("Content-Type", "")
            if not payload:
                raise ValueError(f"Respuesta vacia desde {url}")
            if b"<html" in payload[:500].lower():
                raise ValueError(f"Se recibio HTML en lugar de datos desde {url}")
            temp = destination.with_suffix(destination.suffix + ".tmp")
            temp.write_bytes(payload)
            temp.replace(destination)
            return {
                "url": url,
                "retrieved_at_utc": utc_now(),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "content_type": content_type,
            }
        except Exception as exc:  # pragma: no cover - depende de la red
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"No se pudo descargar {url}: {last_error}")


def max_trailing_run(values) -> tuple[int, object]:
    if len(values) == 0:
        return 0, None
    last = values.iloc[-1]
    count = 0
    for value in reversed(values.tolist()):
        if value != last:
            break
        count += 1
    return count, last
