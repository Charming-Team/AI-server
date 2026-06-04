from fastapi import APIRouter

from app.api.v1.routes import health
from app.api.v1.routes.business_report import router as business_report_router
from app.api.v1.routes.delay_prediction import router as delay_prediction_router
from app.api.v1.routes.report import router as report_router
from app.features.chat.router import router as chat_router

api_router = APIRouter()
api_router.include_router(chat_router)
api_router.include_router(health.router, tags=["health"])
api_router.include_router(report_router)
api_router.include_router(business_report_router)
api_router.include_router(delay_prediction_router)
