"""Delete every Qdrant point whose payload.source_url == the given URL.

Paired with scripts/smoke_ingestion.py — run it when you want the smoke
artifacts out of the collection so they don't show up in eval / demo runs.

Usage:
    uv run python scripts/smoke_cleanup.py https://dev.to/ben/welcome-to-dev-2877
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.ingestion.service import current_collection_name


async def main(url: str) -> int:
    settings = Settings()
    if not settings.qdrant_url or not settings.qdrant_api_key:
        print("QDRANT_URL / QDRANT_API_KEY missing in .env", file=sys.stderr)
        return 1

    client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    collection = current_collection_name()
    url_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="source_url", match=qmodels.MatchValue(value=url))]
    )
    try:
        before = (await client.count(collection_name=collection, count_filter=url_filter)).count
        if before == 0:
            print(f"Nothing to delete — no points for {url!r} in {collection}.")
            return 0
        await client.delete(
            collection_name=collection,
            points_selector=qmodels.FilterSelector(filter=url_filter),
        )
        after = (await client.count(collection_name=collection, count_filter=url_filter)).count
        print(f"Deleted {before - after} / {before} points for {url!r} from {collection}.")
        return 0 if after == 0 else 1
    finally:
        await client.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: smoke_cleanup.py <source_url>", file=sys.stderr)
        sys.exit(2)
    sys.exit(asyncio.run(main(sys.argv[1])))
