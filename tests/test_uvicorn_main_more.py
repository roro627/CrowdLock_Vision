from __future__ import annotations

import runpy
import sys
import types


def test_main_runs_uvicorn_when_executed_as_script(monkeypatch):
    called = {"args": None, "kwargs": None}

    uvicorn_mod = types.ModuleType("uvicorn")

    def _run(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs

    uvicorn_mod.run = _run  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_mod)
    sys.modules.pop("backend.api.main", None)

    runpy.run_module("backend.api.main", run_name="__main__")

    assert called["kwargs"]["host"] == "0.0.0.0"
    assert called["kwargs"]["port"] == 8000
    assert called["kwargs"]["reload"] is True
