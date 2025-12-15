#!/usr/bin/env python3
"""Run backend API and web dev server with a single command."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
CTRL_BREAK_EVENT = getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM)


@dataclass
class ManagedProcess:
    name: str
    handle: subprocess.Popen


class ProcessManager:
    def __init__(self, separate_consoles: bool) -> None:
        self.processes: list[ManagedProcess] = []
        self.separate_consoles = separate_consoles

    def start(
        self, name: str, args: list[str], cwd: Path, env: dict[str, str] | None = None
    ) -> None:
        creationflags = CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        if os.name == "nt" and self.separate_consoles:
            creationflags |= CREATE_NEW_CONSOLE
        process = subprocess.Popen(
            args,
            cwd=str(cwd),
            env=env,
            creationflags=creationflags,
        )
        self.processes.append(ManagedProcess(name=name, handle=process))

    def monitor(self) -> None:
        while True:
            for managed in list(self.processes):
                return_code = managed.handle.poll()
                if return_code is not None:
                    raise RuntimeError(f"{managed.name} exited with code {return_code}")
            time.sleep(0.25)

    def shutdown(self) -> None:
        for managed in self.processes:
            terminate_process(managed)
        for managed in self.processes:
            try:
                managed.handle.wait(timeout=10)
            except subprocess.TimeoutExpired:
                managed.handle.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start backend and frontend dev servers")
    parser.add_argument(
        "--backend-mode",
        choices=("docker", "local"),
        default="docker",
        help="Run backend via docker compose (default) or local uvicorn",
    )
    parser.add_argument("--backend-host", default="0.0.0.0")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-host", default="0.0.0.0")
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument(
        "--skip-backend", action="store_true", help="Do not start the backend service"
    )
    parser.add_argument(
        "--skip-frontend", action="store_true", help="Do not start the frontend dev server"
    )
    parser.add_argument("--backend-app", default="backend.api.main:app", help="Uvicorn app path")
    parser.add_argument(
        "--uvicorn-cmd", default=os.environ.get("UVICORN_CMD", "uvicorn"), help="Uvicorn executable"
    )
    parser.add_argument(
        "--compose-command",
        default=os.environ.get("DOCKER_COMPOSE_CMD", "docker compose"),
        help="Command prefix to run docker compose (e.g. 'docker compose' or 'docker-compose')",
    )
    parser.add_argument(
        "--docker-service", default="backend", help="Docker compose service name for backend"
    )
    parser.add_argument(
        "--docker-build",
        dest="docker_build",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Add --build when starting the backend docker service",
    )
    parser.add_argument(
        "--npm-cmd", default=os.environ.get("NPM_CMD", "npm"), help="npm executable"
    )
    parser.add_argument(
        "--frontend-script", default="dev", help="npm script to run for the frontend"
    )
    parser.add_argument(
        "--auto-install-frontend",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Automatically run npm install if node_modules is missing",
    )
    parser.add_argument(
        "--reload",
        dest="reload",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Toggle uvicorn reload",
    )
    parser.add_argument(
        "--separate-consoles",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Open backend/frontend in independent console windows (Windows only)",
    )
    args = parser.parse_args()
    if args.skip_backend and args.skip_frontend:
        parser.error("Nothing to run: both backend and frontend were skipped")
    return args


def ensure_command_available(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise SystemExit(f"Required command '{executable}' is not available on PATH")
    return resolved


def parse_compose_command(raw: str) -> list[str]:
    parts = shlex.split(raw)
    if not parts:
        raise SystemExit("Compose command cannot be empty")
    return parts


def ensure_frontend_dependencies(frontend_dir: Path, npm_cmd: str) -> None:
    if (frontend_dir / "node_modules").exists():
        return
    print("[setup] node_modules missing, running npm install ...", flush=True)
    subprocess.run([npm_cmd, "install"], cwd=str(frontend_dir), check=True)


def extend_pythonpath(root: Path, current: str | None) -> str:
    if current:
        return f"{root}{os.pathsep}{current}"
    return str(root)


def terminate_process(managed: ManagedProcess) -> None:
    process = managed.handle
    if process.poll() is not None:
        return
    print(f"[shutdown] stopping {managed.name} ...", flush=True)
    try:
        if os.name == "nt":
            process.send_signal(CTRL_BREAK_EVENT)
        else:
            process.terminate()
    except Exception:
        process.kill()


def main() -> None:
    args = parse_args()
    # Go two levels up: scripts/dev/start_stack.py -> <repo>/scripts/dev -> <repo>/scripts -> <repo>
    root = Path(__file__).resolve().parents[2]
    frontend_dir = root / "web"
    manager = ProcessManager(separate_consoles=args.separate_consoles)

    try:
        npm_cmd: str | None = None

        if not args.skip_backend:
            if args.backend_mode == "docker":
                compose_cmd = parse_compose_command(args.compose_command)
                compose_cmd[0] = ensure_command_available(compose_cmd[0])
                backend_cmd = compose_cmd + ["up"]
                if args.docker_build:
                    backend_cmd.append("--build")
                backend_cmd.append(args.docker_service)
                manager.start("backend", backend_cmd, cwd=root)
            else:
                uvicorn_cmd = ensure_command_available(args.uvicorn_cmd)
                backend_env = os.environ.copy()
                backend_env["PYTHONPATH"] = extend_pythonpath(root, backend_env.get("PYTHONPATH"))
                backend_cmd = [
                    uvicorn_cmd,
                    args.backend_app,
                    "--host",
                    args.backend_host,
                    "--port",
                    str(args.backend_port),
                ]
                if args.reload:
                    backend_cmd.extend(
                        [
                            "--reload",
                            "--reload-dir",
                            "backend",
                            "--reload-dir",
                            "config",
                        ]
                    )
                manager.start("backend", backend_cmd, cwd=root, env=backend_env)

        if not args.skip_frontend:
            npm_cmd = ensure_command_available(args.npm_cmd)
            if args.auto_install_frontend:
                ensure_frontend_dependencies(frontend_dir, npm_cmd)
            frontend_cmd = [
                npm_cmd,
                "run",
                args.frontend_script,
                "--",
                "--host",
                args.frontend_host,
                "--port",
                str(args.frontend_port),
            ]
            manager.start("frontend", frontend_cmd, cwd=frontend_dir)

        manager.monitor()
    except KeyboardInterrupt:
        print("\n[info] Received keyboard interrupt, shutting down ...", flush=True)
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main()
