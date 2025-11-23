from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import config, health, stats, stream

app = FastAPI(title="CrowdLock Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(config.router)
app.include_router(stats.router)
app.include_router(stream.router)


@app.on_event("startup")
def startup_event():
    from backend.api.services.state import get_engine
    print("Starting Video Engine...")
    get_engine()

@app.on_event("shutdown")
def shutdown_event():
    from backend.api.services.state import _engine

    if _engine:
        _engine.stop()


if __name__ == "__main__":
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)

