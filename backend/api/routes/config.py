from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.schemas.models import ConfigSchema
from backend.api.services.state import get_settings, reload_settings

router = APIRouter()


@router.get("/config", response_model=ConfigSchema)
def get_config():
    settings = get_settings()
    return ConfigSchema(**settings.dict())


@router.post("/config", response_model=ConfigSchema)
def update_config(cfg: ConfigSchema):
    # Overwrite in-memory settings; persisting to disk is left to env/config files.
    settings = reload_settings(cfg.dict())
    return ConfigSchema(**settings.dict())

