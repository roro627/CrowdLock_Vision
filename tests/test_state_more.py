from __future__ import annotations

import pytest

import backend.api.services.state as state
from backend.core.config.settings import BackendSettings


class DummyEngine:
    def __init__(self, settings: BackendSettings):
        self.settings = settings
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


def test_get_settings_initializes_once(monkeypatch: pytest.MonkeyPatch):
    state._settings = None
    state._engine = None

    calls = {"n": 0}

    def _load():
        calls["n"] += 1
        return BackendSettings(grid_size="10x10")

    monkeypatch.setattr(state, "load_settings", _load)

    s1 = state.get_settings()
    s2 = state.get_settings()
    assert s1.grid_size == "10x10"
    assert s2.grid_size == "10x10"
    assert calls["n"] == 1


def test_reload_settings_recreates_engine_when_running(monkeypatch: pytest.MonkeyPatch):
    state._settings = BackendSettings(grid_size="10x10")
    old_engine = DummyEngine(state._settings)
    state._engine = old_engine

    monkeypatch.setattr(state, "VideoEngine", DummyEngine)
    monkeypatch.setattr(state, "load_settings", lambda: BackendSettings(grid_size="10x10"))

    updated = state.reload_settings({"grid_size": "2x3"})
    assert updated.grid_size == "2x3"
    assert state._engine is not None
    assert old_engine.stopped == 1
    assert state._engine.started == 1


def test_stop_engine_stops_and_clears_engine():
    old_engine = DummyEngine(BackendSettings(grid_size="10x10"))
    state._engine = old_engine
    state.stop_engine()
    assert old_engine.stopped == 1
    assert state._engine is None


def test_reload_settings_without_patch_keeps_defaults(monkeypatch: pytest.MonkeyPatch):
    state._settings = None
    state._engine = None

    monkeypatch.setattr(state, "load_settings", lambda: BackendSettings(grid_size="4x5"))
    updated = state.reload_settings(None)
    assert updated.grid_size == "4x5"


def test_get_engine_creates_and_starts(monkeypatch: pytest.MonkeyPatch):
    state._settings = BackendSettings(grid_size="10x10")
    state._engine = None

    monkeypatch.setattr(state, "VideoEngine", DummyEngine)

    eng = state.get_engine()
    assert isinstance(eng, DummyEngine)
    assert eng.started == 1
