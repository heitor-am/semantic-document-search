from httpx import AsyncClient


async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "commit" in body
    assert body["db"] == "ok"


async def test_root_returns_landing_page(client: AsyncClient) -> None:
    response = await client.get("/")
    assert response.status_code == 200
    assert "Semantic Document Search" in response.text
