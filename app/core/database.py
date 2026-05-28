from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.core.config import get_settings

settings = get_settings()

engine: Engine = create_engine(
    settings.report_postgres_url,
    pool_pre_ping=True,
)
