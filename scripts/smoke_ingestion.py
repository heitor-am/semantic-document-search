"""End-to-end smoke test against real Qdrant + OpenRouter + dev.to.

Why this exists
---------------
Unit tests use respx / MagicMock / fake vector repos. They prove the code
*within* our process does the right thing, but they can't prove that:

- baai/bge-m3 via OpenRouter actually returns 1024-dim vectors (the
  collection is provisioned for that size; a mismatch blows up on upsert).
- Qdrant Cloud accepts our ensure_collection + payload-index calls and
  responds in the shape qdrant-client expects.
- dev.to's real `body_markdown` renders into a sensible chunk set.

Run this before trusting the ingestion pipeline end-to-end.

Usage
-----
    # Terminal 1 — start the app against real deps:
    cp .env.example .env   # fill QDRANT_URL, QDRANT_API_KEY, OPENROUTER_API_KEY
    make migrate
    make dev

    # Terminal 2 — run the smoke:
    make smoke
    # or: uv run python scripts/smoke_ingestion.py [SMOKE_URL]

What it checks
--------------
1. Preflight — required env vars are set; embedding dim is registered.
2. /health responds 200.
3. POST /ingest → 202 with the expected deterministic job_id.
4. Job transitions through fetch→parse→chunk→embed→index and lands in
   COMPLETED within a 60s timeout.
5. Qdrant count(source_url=URL) > 0, with both parents and children.
6. First point payload carries the canonical keys
   (source_url, source_type, is_parent, section_path, chunk_index, title).
7. Re-POST same URL → same job_id, 202, state=completed; Qdrant count
   unchanged (upsert idempotency via deterministic chunk_ids).

Cost: one dev.to post ingests a handful of chunks; baai/bge-m3 on
OpenRouter bills a few cents for that.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

# Allow `uv run python scripts/smoke_ingestion.py` without install.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.ingestion.service import current_collection_name, deterministic_job_id
from app.shared.qdrant.collections import vector_size_for

APP_URL = "http://127.0.0.1:8000"
DEFAULT_SMOKE_URL = "https://dev.to/ben/welcome-to-dev-2877"


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")


def _ok(msg: str) -> None:
    print(f"  OK:   {msg}")


async def _preflight(settings: Settings) -> bool:
    print("=== 1. Preflight ===")
    if not settings.qdrant_url or not settings.qdrant_api_key:
        _fail("QDRANT_URL / QDRANT_API_KEY missing in .env")
        return False
    if not settings.openrouter_api_key:
        _fail("OPENROUTER_API_KEY missing in .env")
        return False
    try:
        dim = vector_size_for(settings.openrouter_embedding_model)
    except ValueError as exc:
        _fail(str(exc))
        return False
    _ok(f"Qdrant: {settings.qdrant_url}")
    _ok(f"OpenRouter: {settings.openrouter_base_url}")
    _ok(f"Embedding model: {settings.openrouter_embedding_model} ({dim} dims)")
    _ok(f"Collection: {current_collection_name()}")
    return True


async def _check_health(client: httpx.AsyncClient) -> bool:
    print("\n=== 2. App health ===")
    try:
        r = await client.get(f"{APP_URL}/health", timeout=5.0)
    except httpx.RequestError as exc:
        _fail(f"App not reachable at {APP_URL} (is `make dev` running?): {exc}")
        return False
    if r.status_code != 200:
        _fail(f"/health returned {r.status_code}: {r.text}")
        return False
    _ok(f"/health 200: {r.json()}")
    return True


async def _post_ingest(client: httpx.AsyncClient, url: str) -> str | None:
    print("\n=== 3. POST /ingest ===")
    r = await client.post(f"{APP_URL}/ingest", json={"source_url": url}, timeout=30.0)
    if r.status_code != 202:
        _fail(f"expected 202, got {r.status_code}: {r.text}")
        return None
    body = r.json()
    expected = deterministic_job_id(url)
    if body["job_id"] != expected:
        _fail(f"job_id {body['job_id']} != deterministic hash {expected}")
        return None
    _ok(f"202 Accepted, job_id={body['job_id']}, state={body['state']}")
    return body["job_id"]


async def _poll_until_terminal(
    client: httpx.AsyncClient,
    job_id: str,
    timeout_s: int = 60,
) -> dict | None:
    print("\n=== 4. Poll for terminal state ===")
    for attempt in range(timeout_s):
        r = await client.get(f"{APP_URL}/ingest/jobs/{job_id}", timeout=5.0)
        body = r.json()
        state = body["state"]
        if state in ("completed", "failed"):
            _ok(f"terminal state after {attempt + 1}s: {state}")
            history = body.get("history", [])
            _ok("transitions: " + " → ".join(h["state"] for h in history))
            if state == "failed":
                _fail(f"ingestion failed with error: {body.get('error')}")
                return None
            return body
        await asyncio.sleep(1)
    _fail(f"job did not reach a terminal state in {timeout_s}s")
    return None


async def _verify_qdrant(
    qclient: AsyncQdrantClient,
    collection: str,
    url: str,
) -> int | None:
    print("\n=== 5. Verify Qdrant state ===")

    url_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="source_url", match=qmodels.MatchValue(value=url))]
    )
    total = (await qclient.count(collection_name=collection, count_filter=url_filter)).count
    if total < 1:
        _fail("no points stored for this URL")
        return None
    _ok(f"total points for URL: {total}")

    parents = (
        await qclient.count(
            collection_name=collection,
            count_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="source_url", match=qmodels.MatchValue(value=url)),
                    qmodels.FieldCondition(key="is_parent", match=qmodels.MatchValue(value=True)),
                ],
            ),
        )
    ).count
    children = total - parents
    _ok(f"parents={parents}, children={children}")
    if parents == 0 or children == 0:
        _fail("expected at least 1 parent and 1 child")
        return None

    scroll_points, _ = await qclient.scroll(
        collection_name=collection,
        scroll_filter=url_filter,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not scroll_points:
        _fail("scroll returned nothing")
        return None
    payload = scroll_points[0].payload or {}
    required = {"source_url", "source_type", "is_parent", "section_path", "chunk_index", "title"}
    missing = required - payload.keys()
    if missing:
        _fail(f"missing payload keys: {missing}")
        return None
    _ok(f"payload shape OK: {sorted(payload.keys())}")
    _ok(f"sample: title={payload['title']!r}, is_parent={payload['is_parent']}")
    return total


async def _verify_idempotent_repost(
    client: httpx.AsyncClient,
    qclient: AsyncQdrantClient,
    collection: str,
    url: str,
    job_id: str,
    count_before: int,
) -> bool:
    print("\n=== 6. Re-POST same URL (idempotency) ===")
    r = await client.post(f"{APP_URL}/ingest", json={"source_url": url}, timeout=10.0)
    body = r.json()
    if body["job_id"] != job_id:
        _fail(f"job_id changed on re-POST: {body['job_id']} != {job_id}")
        return False
    if body["state"] != "completed":
        _fail(f"expected completed (short-circuit), got {body['state']}")
        return False
    _ok("same job_id, state=completed (short-circuited)")

    url_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="source_url", match=qmodels.MatchValue(value=url))]
    )
    total_after = (await qclient.count(collection_name=collection, count_filter=url_filter)).count
    if total_after != count_before:
        _fail(f"point count changed after re-POST: {count_before} → {total_after}")
        return False
    _ok(f"point count stable at {total_after}")
    return True


async def main(source_url: str) -> int:
    settings = Settings()

    if not await _preflight(settings):
        return 1

    async with httpx.AsyncClient() as http:
        if not await _check_health(http):
            return 1

        job_id = await _post_ingest(http, source_url)
        if job_id is None:
            return 1

        result = await _poll_until_terminal(http, job_id)
        if result is None:
            return 1

        qclient = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        try:
            count = await _verify_qdrant(qclient, current_collection_name(), source_url)
            if count is None:
                return 1
            if not await _verify_idempotent_repost(
                http, qclient, current_collection_name(), source_url, job_id, count
            ):
                return 1
        finally:
            await qclient.close()

    print("\n=== ALL SMOKE CHECKS PASSED ===")
    print("\nCleanup (optional): delete this URL's points from the collection —")
    print(f"  uv run python scripts/smoke_cleanup.py {source_url!r}")
    return 0


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SMOKE_URL
    sys.exit(asyncio.run(main(url)))
