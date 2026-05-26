from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.api.v1.routes import health
from app.core.config import get_settings
from app.features.chat.error_handlers import register_chat_error_handlers

AI_HEALTH_ALIAS_PREFIX = "/ai"


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_chat_error_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    if _should_register_ai_health_alias(settings.api_v1_prefix):
        app.include_router(
            health.router,
            prefix=AI_HEALTH_ALIAS_PREFIX,
            tags=["health"],
            include_in_schema=False,
        )

    @app.get("/", tags=["root"])
    async def root() -> dict[str, str]:
        return {"message": settings.app_name, "docs": "/docs"}

    return app


def _should_register_ai_health_alias(api_v1_prefix: str) -> bool:
    normalized_prefix = api_v1_prefix.rstrip("/")
    return (
        normalized_prefix != AI_HEALTH_ALIAS_PREFIX
        and normalized_prefix.startswith(f"{AI_HEALTH_ALIAS_PREFIX}/")
    )


app = create_app()
