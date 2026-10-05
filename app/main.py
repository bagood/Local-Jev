from fastapi import FastAPI

from app.routers.inference import router as inference_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local JEV API",
        description="Choice inference using local next-token logits.",
        version="1.0.0",
    )
    app.include_router(inference_router)

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
