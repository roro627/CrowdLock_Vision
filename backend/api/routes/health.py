"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Lightweight health endpoint used by containers and dev tooling."""

    return {"status": "ok"}
