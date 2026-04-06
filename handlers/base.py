from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from config import ModelConfig


class BaseHandler(ABC):
    def __init__(self, base_url: str, http_client: httpx.AsyncClient) -> None:
        self.base_url = base_url
        self.client = http_client

    @abstractmethod
    async def handle(
        self,
        request: Request,
        path: str,
        body: dict,
        model_config: ModelConfig,
    ) -> tuple[Union[Response, StreamingResponse], Optional[dict]]:
        """
        Handle the request and return (response, sent_body).
        sent_body is the request body as actually sent to the backend,
        or None if the original body was sent unchanged.
        """
        ...
