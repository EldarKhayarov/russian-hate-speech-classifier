import os

import pytest
from httpx import AsyncClient


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def http_client():
    api_host = os.environ.get("API_URL", "http://localhost:8000")
    async with AsyncClient(base_url=api_host) as client:
        yield client


@pytest.mark.anyio
async def test_prediction(http_client):
    response = await http_client.post(
        "/api/v1/predict",
        json={
            "comment": "Хохлы, это отдушина затюканого россиянина, мол, вон, а у хохлов еще хуже. "
            "Если бы хохлов не было, кисель их бы придумал."
        },
    )
    assert response.status_code == 200, response.json()
    assert response.json()["toxic"] == 1

    response = await http_client.post(
        "/api/v1/predict",
        json={
            "comment": "В шапке были ссылки на инфу по текущему фильму марвел. "
            "Эти ссылки были заменены на фразу Репортим брипидора, игнорируем его посты. "
            "Если этого недостаточно, чтобы понять, что модератор абсолютный неадекват, и его нужно лишить "
            "полномочий, тогда эта борда пробивает абсолютное дно по неадекватности."
        },
    )
    assert response.status_code == 200, response.json()
    assert response.json()["toxic"] == 0
