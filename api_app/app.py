import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_app.settings import get_app_settings
from api_app.api import router as api_router
from api_app.exceptions import attribute_error_handler


def get_application():
    settings = get_app_settings()

    application = FastAPI(**settings.fastapi_kwargs)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_hosts,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    application.add_exception_handler(AttributeError, attribute_error_handler)
    application.include_router(api_router, prefix=settings.api_prefix)

    return application


app = get_application()


if __name__ == "__main__":
    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
