from typing import Any

from sqlalchemy import text

from app.core.database import engine
from app.features.delay_prediction.features import FEATURE_COLUMNS

VIEW_NAME = "delay_prediction_evidence.vw_delay_prediction_inference_orders"


class DelayPredictionRepository:
    def fetch_inference_row(self, order_id: int) -> dict[str, Any] | None:
        select_columns = ", ".join(["order_id", *FEATURE_COLUMNS])
        query = text(
            f"""
            SELECT {select_columns}
            FROM {VIEW_NAME}
            WHERE order_id = :order_id
            LIMIT 1
            """
        )

        with engine.connect() as conn:
            row = conn.execute(query, {"order_id": order_id}).mappings().first()

        if row is None:
            return None

        return dict(row)
