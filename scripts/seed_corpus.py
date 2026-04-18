"""Bulk-ingest the URLs in scripts/seed_urls.txt through the running /ingest endpoint.

Why not a one-shot bulk endpoint? The FSM-driven, per-job model is the
product — this script intentionally drives it the same way an operator
would, so we also prove the endpoint handles sustained throughput.

Usage
-----
    # Terminal 1 — the app must be running against real Qdrant + OpenRouter
    make migrate && make dev

    # Terminal 2
    make seed                               # reads scripts/seed_urls.txt
    # or a custom file:
    uv run python scripts/seed_corpus.py path/to/urls.txt

Behaviour
---------
- Reads URLs from the file, skipping blank lines and `#`-comments.
- For each URL: POSTs /ingest, then polls /ingest/jobs/{id} until the
  job reaches a terminal state (completed / failed), up to 60s.
- Runs sequentially — keeps the load predictable and the Qdrant collection
  consistent. Parallelism would complicate rate-limit handling against
  dev.to + OpenRouter for little wall-clock gain at this scale.
- Prints a per-URL line (`OK` / `FAIL`) and a final summary.
- Poll timeout is POLL_TIMEOUT_S iterations at POLL_INTERVAL_S each
  (90s by default — bge-m3 embed + index for ~25 chunks is the worst
  case).

Exit code: 0 if every URL completed, 1 if any failed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APP_URL = "http://127.0.0.1:8000"
POLL_TIMEOUT_S = 90  # bge-m3 embed + rerank call for ~25 chunks per post
POLL_INTERVAL_S = 1.0


def read_urls(path: Path) -> list[str]:
    urls: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


async def ingest_one(client: httpx.AsyncClient, url: str) -> tuple[str, str, str | None]:
    """Returns (state, job_id, error)."""
    try:
        r = await client.post(f"{APP_URL}/ingest", json={"source_url": url}, timeout=15.0)
    except httpx.RequestError as exc:
        return ("failed", "-", f"POST failed: {exc}")
    if r.status_code != 202:
        return ("failed", "-", f"POST returned {r.status_code}: {r.text[:100]}")

    body = r.json()
    job_id = body["job_id"]
    if body["state"] == "completed":
        # Idempotent short-circuit — already ingested previously.
        return ("completed", job_id, None)

    # Poll until terminal
    for _ in range(POLL_TIMEOUT_S):
        await asyncio.sleep(POLL_INTERVAL_S)
        try:
            r = await client.get(f"{APP_URL}/ingest/jobs/{job_id}", timeout=5.0)
        except httpx.RequestError:
            continue
        if r.status_code != 200:
            continue
        state = r.json()["state"]
        if state in ("completed", "failed"):
            err = r.json().get("error")
            return (state, job_id, err)
    return ("timeout", job_id, f"did not reach terminal state in {POLL_TIMEOUT_S}s")


async def main(urls_path: Path) -> int:
    urls = read_urls(urls_path)
    if not urls:
        print(f"No URLs in {urls_path}", file=sys.stderr)
        return 1

    print(f"Seeding {len(urls)} URLs via {APP_URL}/ingest ...")
    print()

    failures: list[tuple[str, str]] = []
    completed = 0
    async with httpx.AsyncClient() as client:
        # Verify the app is up before the first POST
        try:
            r = await client.get(f"{APP_URL}/health", timeout=5.0)
            r.raise_for_status()
        except Exception as exc:
            print(f"FAIL: app not reachable at {APP_URL} ({exc}). Is `make dev` running?")
            return 1

        for i, url in enumerate(urls, 1):
            state, _job_id, err = await ingest_one(client, url)
            if state == "completed":
                completed += 1
                print(f"  [{i:2d}/{len(urls)}] OK   {url}")
            else:
                failures.append((url, err or state))
                print(f"  [{i:2d}/{len(urls)}] FAIL {url}  ({state}: {err})")

    print()
    print(f"Summary: {completed}/{len(urls)} ingested")
    if failures:
        print()
        print("Failures:")
        for url, err in failures:
            print(f"  - {url}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "scripts" / "seed_urls.txt"
    sys.exit(asyncio.run(main(path)))
