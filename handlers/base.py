from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

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
    ) -> Union[Response, StreamingResponse]:
        ...
