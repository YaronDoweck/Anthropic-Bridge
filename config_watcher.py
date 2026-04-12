"""
ConfigWatcher: polls proxy.config for file changes and hot-reloads the router.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable, Optional

import httpx

from config import ProxyConfig

logger = logging.getLogger(__name__)

CONFIG_POLL_INTERVAL_S = 2.0


class ConfigWatcher:
    def __init__(
        self,
        path: str,
        router,  # ProxyRouter — avoid circular import at class-definition time
        http_client: httpx.AsyncClient,
        apply_env_overrides: Callable[[ProxyConfig], None],
        interval_s: float = CONFIG_POLL_INTERVAL_S,
    ) -> None:
        self._path = path
        self._router = router
        self._http_client = http_client
        self._apply_env_overrides = apply_env_overrides
        self._interval_s = interval_s
        self._stop_event = asyncio.Event()
        self._last_stat: Optional[os.stat_result] = None

    async def run(self) -> None:
        """Main polling loop. Exits when stop() is called."""
        # Seed the last-known stat from the currently-loaded file
        try:
            self._last_stat = await asyncio.get_running_loop().run_in_executor(
                None, os.stat, self._path
            )
        except OSError:
            pass

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._interval_s
                )
            except asyncio.TimeoutError:
                pass  # Normal — timeout means we should poll

            if self._stop_event.is_set():
                break

            try:
                stat = await asyncio.get_running_loop().run_in_executor(
                    None, os.stat, self._path
                )
            except OSError as exc:
                logger.warning("ConfigWatcher: cannot stat %r: %s", self._path, exc)
                continue

            changed = (
                self._last_stat is None
                or stat.st_mtime_ns != self._last_stat.st_mtime_ns
                or stat.st_size != self._last_stat.st_size
            )
            if changed:
                await self._reload(stat)

    async def _reload(self, new_stat: os.stat_result) -> None:
        from config import load_config  # local import to avoid circular issues at module level

        logger.info("ConfigWatcher: detected change in %r, reloading…", self._path)
        try:
            new_config = load_config(self._path)
            self._apply_env_overrides(new_config)
        except Exception as exc:
            logger.error(
                "ConfigWatcher: reload failed, keeping previous config: %s", exc
            )
            return

        # Validate override_model if present
        override = new_config.server.override_model
        if override and override not in new_config.models:
            logger.error(
                "ConfigWatcher: reload aborted — override_model '%s' not in new config models. "
                "Keeping previous config.",
                override,
            )
            return

        try:
            await self._router.apply_config(new_config, self._http_client)
        except Exception as exc:
            logger.error(
                "ConfigWatcher: apply_config failed, keeping previous config: %s", exc
            )
            return

        self._last_stat = new_stat
        logger.info("ConfigWatcher: config reloaded successfully")

    def stop(self) -> None:
        """Signal the watcher loop to exit."""
        self._stop_event.set()
