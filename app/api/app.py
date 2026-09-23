from fastapi import FastAPI
from app.api.routes.background_removal import (
    router as background_removal_router,
)


app = FastAPI(
    title="AI Background Removal API",
    description=(
        "Production API for AI-powered "
        "image background removal using BiRefNet."
    ),
    version="1.0.0",
)


app.include_router(
    background_removal_router
)


@app.get(
    "/",
    tags=["Health"],
)
def root():
    return {
        "service": "AI Background Removal API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get(
    "/health",
    tags=["Health"],
)
def health():
    return {
        "status": "healthy",
    }