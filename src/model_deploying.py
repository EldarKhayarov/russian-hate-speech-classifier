import asyncio
import os
import json
import sys
import logging
from typing import IO

from httpx import AsyncClient


ROOT_DIR = ""
DATA_DIR = os.path.join(ROOT_DIR, "data")
FACTORIZER_PATH = os.path.join(ROOT_DIR, "factorizer/tfidf_vectorizer.pickle")
RESULTS_PATH = os.path.join(DATA_DIR, "evaluation-storage/results.json")
BEST_MODEL_PATH_FMT = os.path.join(
    DATA_DIR, "train-storage/{model_name}/best__model.pickle"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_deploying")


def get_best_model_name() -> str:
    results = json.load(open(RESULTS_PATH))
    return results["__BEST_MODEL"]["model_name"]


def get_best_model_file() -> IO:
    return open(BEST_MODEL_PATH_FMT.format(model_name=get_best_model_name()), "rb")


def get_factorizer_file() -> IO:
    return open(FACTORIZER_PATH, "rb")


def upload_factorizer_coro(client: AsyncClient):
    logger.info("Uploading factorizer...")
    return client.post(
        "/api/v1/upload-factorizer",
        files={"factorizer_upload": ("factorizer.pickle", get_factorizer_file())},
    )


def upload_model_coro(client: AsyncClient):
    logger.info("Uploading model...")
    return client.post(
        "/api/v1/upload-model",
        files={"model_upload": ("model.pickle", get_best_model_file())},
    )


async def deploy_model(api_address: str):
    logger.info("Connecting to server " + api_address)
    async with AsyncClient(base_url=api_address, timeout=10000) as client:
        rsp_factorizer, rsp_model = await asyncio.gather(
            upload_factorizer_coro(client),
            upload_model_coro(client),
        )

    # Don't use `assert`!
    if rsp_factorizer.status_code != 200:
        raise AssertionError("Factorizer uploading error")

    if rsp_model.status_code != 200:
        raise AssertionError("Model uploading error")


if __name__ == "__main__":
    asyncio.run(deploy_model(sys.argv[1]))
