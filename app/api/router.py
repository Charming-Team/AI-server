from fastapi import APIRouter

from app.api.v1.routes import health
from app.features.chat.router import router as chat_router
from app.api.v1.routes.report import router as report_router

api_router = APIRouter()
api_router.include_router(chat_router)
api_router.include_router(health.router, tags=["health"])
api_router.include_router(report_router)