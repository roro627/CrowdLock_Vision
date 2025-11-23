from __future__ import annotations

from typing import Optional

from backend.core.config.settings import BackendSettings, load_settings
from backend.api.services.engine import VideoEngine

from threading import RLock

_settings: Optional[BackendSettings] = None
_engine: Optional[VideoEngine] = None
_lock = RLock()


def get_settings() -> BackendSettings:
    global _settings
    with _lock:
        if _settings is None:
            _settings = load_settings()
    return _settings


def reload_settings(data: dict | None = None) -> BackendSettings:
    global _settings, _engine
    with _lock:
        base = load_settings()
        if data:
            _settings = BackendSettings(**{**base.dict(), **data})
        else:
            _settings = base
        if _engine:
            # Recreate engine with new settings
            _engine.stop()
            _engine = VideoEngine(_settings)
            _engine.start()
    return _settings


def get_engine() -> VideoEngine:
    global _engine
    with _lock:
        if _engine is None:
            _engine = VideoEngine(get_settings())
            _engine.start()
    return _engine

