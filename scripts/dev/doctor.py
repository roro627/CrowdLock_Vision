#!/usr/bin/env python3
"""Quick environment & project sanity checks.

Goal: give a great first-run experience (especially on Windows) by checking
common prerequisites and printing actionable next steps.

No external dependencies.
"""

from __future__ import annotations

import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    """Result of a single doctor check."""

    name: str
    ok: bool
    detail: str


def _which(cmd: str) -> str | None:
    """Return the resolved path for a command on PATH (or None)."""

    return shutil.which(cmd)


def _run_version(cmd: list[str]) -> str | None:
    """Run a command and return its version-like output (stdout or stderr)."""

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    text = out if out else err
    return text if text else None


def _port_available(host: str, port: int) -> bool:
    """Return True if the TCP port appears available (best-effort)."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            return s.connect_ex((host, port)) != 0
    except Exception:
        # If we can't check, don't block onboarding.
        return True


def check_commands() -> list[CheckResult]:
    """Check that key developer tools are installed and discoverable."""

    results: list[CheckResult] = []

    python_ver = platform.python_version()
    is_py312 = sys.version_info[:2] == (3, 12)
    py_detail = f"{sys.executable} (Python {python_ver})"
    if not is_py312:
        py_detail += " (expected Python 3.12.x)"
    results.append(CheckResult("python", is_py312, py_detail))

    for cmd in ("node", "npm", "docker"):
        path = _which(cmd)
        if not path:
            results.append(CheckResult(cmd, False, "not found on PATH"))
            continue
        ver = _run_version([cmd, "--version"]) or "(version unknown)"
        results.append(CheckResult(cmd, True, f"{path} ({ver})"))

    # uvicorn is a python package; check via python -m uvicorn
    uv = _run_version([sys.executable, "-m", "uvicorn", "--version"])
    if uv:
        results.append(CheckResult("uvicorn", True, uv))
    else:
        results.append(CheckResult("uvicorn", False, "not installed in this Python env"))

    return results


def check_project_files(root: Path) -> list[CheckResult]:
    """Check for expected repo files/folders used by common workflows."""

    results: list[CheckResult] = []

    cfg = root / "config" / "backend.config.yml"
    cfg_example = root / "config" / "backend.config.example.yml"
    if cfg.exists():
        results.append(CheckResult("config/backend.config.yml", True, "present"))
    elif cfg_example.exists():
        results.append(
            CheckResult(
                "config/backend.config.yml",
                False,
                "missing (copy config/backend.config.example.yml -> config/backend.config.yml)",
            )
        )
    else:
        results.append(
            CheckResult("config/backend.config.yml", False, "missing (and no example found)")
        )

    web_pkg = root / "web" / "package.json"
    results.append(
        CheckResult(
            "web/package.json", web_pkg.exists(), "present" if web_pkg.exists() else "missing"
        )
    )

    node_modules = root / "web" / "node_modules"
    results.append(
        CheckResult(
            "web/node_modules",
            node_modules.exists(),
            "present" if node_modules.exists() else "missing (run: cd web && npm install)",
        )
    )

    videos = root / "testdata" / "videos"
    results.append(
        CheckResult("testdata/videos", videos.exists(), "present" if videos.exists() else "missing")
    )

    return results


def check_ports() -> list[CheckResult]:
    """Check whether the default dev ports appear to be available."""

    checks = [("backend port", "127.0.0.1", 8000), ("web port", "127.0.0.1", 5173)]
    results: list[CheckResult] = []
    for name, host, port in checks:
        ok = _port_available(host, port)
        detail = "available" if ok else f"in use on {host}:{port}"
        results.append(CheckResult(name, ok, detail))
    return results


def _print_section(title: str) -> None:
    """Print a human-readable section header."""

    print(f"\n== {title} ==")


def _print_results(results: list[CheckResult]) -> None:
    """Print a list of check results in a compact format."""

    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")


def main() -> int:
    """Entry point for the doctor CLI."""

    root = Path(__file__).resolve().parents[2]

    print("CrowdLock Vision – doctor")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Repo: {root}")

    _print_section("Commands")
    cmd_results = check_commands()
    _print_results(cmd_results)

    _print_section("Project")
    proj_results = check_project_files(root)
    _print_results(proj_results)

    _print_section("Ports")
    port_results = check_ports()
    _print_results(port_results)

    failed = [r for r in (cmd_results + proj_results + port_results) if not r.ok]

    _print_section("Next steps")
    if failed:
        print("Fix the FAIL items above, then try:")
    else:
        print("Looks good. Try:")

    print("- Backend+web (docker backend):  make dev")
    print("- Backend+web (local backend):   make dev-local")
    print("- Backend tests:                 pytest -q")
    print("- Web tests:                     cd web && npm test")

    # Helpful demo hint.
    demo_video = root / "testdata" / "videos" / "855564-hd_1920_1080_24fps.mp4"
    if demo_video.exists():
        print("\nDemo (file source):")
        print(
            "- Set video_source=file and video_path=testdata/videos/855564-hd_1920_1080_24fps.mp4 in config/backend.config.yml"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
