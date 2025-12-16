from __future__ import annotations

from threading import RLock

from backend.api.services.engine import VideoEngine
from backend.core.config.settings import BackendSettings, load_settings, settings_to_dict

_settings: BackendSettings | None = None
_engine: VideoEngine | None = None
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
            _settings = BackendSettings(**{**settings_to_dict(base), **data})
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


def stop_engine() -> None:
    global _engine
    with _lock:
        if _engine is not None:
            _engine.stop()
            _engine = None
