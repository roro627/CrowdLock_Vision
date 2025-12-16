from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.schemas.models import ConfigSchema
from backend.api.services.state import get_settings, reload_settings
from backend.core.config.settings import settings_to_dict
from backend.core.config.presets import list_presets, preset_patch

router = APIRouter()


@router.get("/config", response_model=ConfigSchema)
def get_config():
    settings = get_settings()
    return ConfigSchema(**settings_to_dict(settings))


@router.get("/config/presets")
def get_presets():
    return {"presets": list_presets()}


@router.post("/config/presets/{preset_id}", response_model=ConfigSchema)
def apply_preset(preset_id: str):
    try:
        patch = preset_patch(preset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown preset")
    settings = reload_settings(patch)
    return ConfigSchema(**settings_to_dict(settings))


@router.post("/config", response_model=ConfigSchema)
def update_config(cfg: ConfigSchema):
    # Overwrite in-memory settings; persisting to disk is left to env/config files.
    data = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    settings = reload_settings(data)
    return ConfigSchema(**settings_to_dict(settings))
