import os.path

import pytest
from fastapi import status
from httpx import AsyncClient

from api_app.app import get_application


HOST = "localhost"
PORT = "8000"

ROOT_DIR = os.environ.get("PROJECT_ROOT_DIR", "..")
FACTORIZER_PATH = os.path.join(ROOT_DIR, "factorizer/tfidf_vectorizer.pickle")
MODEL_PATH = os.path.join(ROOT_DIR, "data/train-storage/svc/best__model.pickle")


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def application():
    yield get_application()


@pytest.fixture
async def http_client(application):
    async with AsyncClient(
        app=application, base_url="http://{host}:{port}".format(host=HOST, port=PORT)
    ) as client:
        yield client


@pytest.fixture
async def factorizer():
    yield open(FACTORIZER_PATH, "rb")


@pytest.fixture
async def model():
    yield open(MODEL_PATH, "rb")


@pytest.mark.anyio
async def test_prediction(http_client, factorizer, model):
    response = await http_client.post(
        "/api/v1/upload-factorizer",
        files={"factorizer_upload": ("factorizer.pickle", factorizer)},
    )
    assert response.status_code == status.HTTP_200_OK

    response = await http_client.post(
        "/api/v1/upload-model",
        files={"model_upload": ("best__model.pickle", model)},
    )
    assert response.status_code == status.HTTP_200_OK

    response = await http_client.post(
        "/api/v1/predict",
        json={
            "comment": "Хохлы, это отдушина затюканого россиянина, мол, вон, а у хохлов еще хуже. "
            "Если бы хохлов не было, кисель их бы придумал."
        },
    )
    assert response.status_code == status.HTTP_200_OK, response.json()
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
    assert response.status_code == status.HTTP_200_OK, response.json()
    assert response.json()["toxic"] == 0
