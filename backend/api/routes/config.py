"""Configuration endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.schemas.models import ConfigSchema
from backend.api.services.state import get_settings, reload_settings
from backend.core.config.presets import list_presets, preset_patch
from backend.core.config.settings import settings_to_dict

router = APIRouter()


@router.get("/config", response_model=ConfigSchema)
def get_config() -> ConfigSchema:
    """Return the current effective backend configuration."""

    settings = get_settings()
    return ConfigSchema(**settings_to_dict(settings))


@router.get("/config/presets")
def get_presets() -> dict[str, list[dict[str, object]]]:
    """Return available runtime presets."""

    return {"presets": list_presets()}


@router.post("/config/presets/{preset_id}", response_model=ConfigSchema)
def apply_preset(preset_id: str) -> ConfigSchema:
    """Apply a preset by id and return the updated configuration."""

    try:
        patch = preset_patch(preset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown preset") from None
    settings = reload_settings(patch)
    return ConfigSchema(**settings_to_dict(settings))


@router.post("/config", response_model=ConfigSchema)
def update_config(cfg: ConfigSchema) -> ConfigSchema:
    """Update in-memory settings and restart the engine.

    This endpoint updates runtime configuration only. Persist configuration via
    environment variables or the YAML config file.
    """

    data = cfg.model_dump()
    settings = reload_settings(data)
    return ConfigSchema(**settings_to_dict(settings))
