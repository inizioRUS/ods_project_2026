from __future__ import annotations

from fastapi import FastAPI

from api.routes import router
from core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app.title, version=settings.app.version)
app.include_router(router)


@app.get("/")
def root() -> dict:
    return {
        "service": settings.app.title,
        "version": settings.app.version,
        "docs": "/docs",
        "health": "/v1/health",
    }
