# Repository Guidelines

## Project Structure & Module Organization

- Current contents: `SPEC.md` (authoritative requirements) and `testdata/videos/` sample clips used for validation.
- Target layout (follow SPEC): `backend/` (core CV, trackers, analytics, API), `web/` (React UI), `scripts/` (dev/build helpers), `config/` (env examples). Keep detectors, trackers, analytics, and video sources in separate subfolders.
- Keep new assets small; large or proprietary footage should stay out of the repo unless explicitly cleared.

## Build, Test, and Development Commands

- Backend (once scaffolded): `python -m backend.api.main` to run the API; prefer a `scripts/dev/start_backend.sh` wrapper for env setup.
- Frontend: from `web/`, `npm install` then `npm run dev` for local UI; `npm run build` for production bundle.
- Docker: add `docker compose up --build` to launch backend + web once compose files exist.

## Coding Style & Naming Conventions

- Python: 4‑space indent, type hints required, `black` + `ruff` for formatting/linting; snake_case modules, PascalCase classes, lower_snake functions/vars.
- JS/TS (web): `prettier` + `eslint` with TypeScript strict; React components PascalCase, hooks `useThing`, files kebab-case or component name.
- Config: `.env` for secrets (never commit); YAML/JSON configs live in `config/` with `.example` templates.

## Testing Guidelines

- Backend: `pytest` with >80% coverage target for analytics math; fixture videos come from `testdata/videos/`—keep tests deterministic (cap FPS, fixed seeds).
- Web: `npm run test` (Vitest/RTL) for components; snapshot overlays sparingly.
- Name tests `test_*.py` and `*.spec.tsx`; include at least one regression test per bug fix.

## Commit & Pull Request Guidelines

- Only initial commit; adopt Conventional Commits (e.g., `feat: add density grid`, `fix: handle empty frame`).
- PRs: concise description, linked issue, testing notes (`pytest`, `npm test`, `docker compose up` if relevant), and screenshots/GIFs for UI or overlay changes.
- Keep PRs small and focused; include docs updates when adding commands or configs.

## Security & Configuration Tips

- Do not hardcode credentials or stream URLs; load via environment and document in `config/*.example`.
- Avoid committing large raw videos; reference paths under `testdata/videos/` or provide download instructions.
- Enable logging but redact PII; when exporting analytics, keep outputs anonymous (IDs only).
