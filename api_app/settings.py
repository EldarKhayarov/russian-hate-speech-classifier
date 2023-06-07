from typing import Dict, Any

from pydantic import BaseSettings


class AppSettings(BaseSettings):
    """
    Класс для хранения констант конфига и удобного к ним доступа из разных участков кода.
    """

    debug: bool = False
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    openapi_prefix: str = ""
    openapi_url: str = "/openapi.json"
    redoc_url: str = "/redoc"
    title: str = "Russian hate speech ML model [FastAPI + Python 3.10]"
    version: str = "0.0.1"

    allowed_hosts: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    @property
    def fastapi_kwargs(self) -> Dict[str, Any]:
        return {
            "debug": self.debug,
            "docs_url": self.docs_url,
            "openapi_prefix": self.openapi_prefix,
            "openapi_url": self.openapi_url,
            "redoc_url": self.redoc_url,
            "title": self.title,
            "version": self.version,
        }


def get_app_settings():
    return AppSettings()
