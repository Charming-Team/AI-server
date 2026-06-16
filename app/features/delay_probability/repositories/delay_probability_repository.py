# app/features/delay_probability/repositories/delay_probability_repository.py
"""
Repository for delay probability inference source rows.

역할:
- inference 전용 DB view에서 order_id 기준으로 row 1건 조회
- DB 조회만 담당
- 모델 전처리, 추론, 응답 생성은 service/artifact 계층에서 처리

중요:
- 이 기능은 plan별 예측이 아니라 주문별 지연 확률 예측입니다.
- 따라서 inference view는 order_id당 1 row를 반환해야 합니다.
- 주문에 연결된 여러 plan 정보는 view 내부에서 주문 단위로 집계되어야 합니다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.database import engine as default_engine
from app.features.delay_probability.preprocess import (
    get_required_inference_view_columns,
)

DEFAULT_INFERENCE_VIEW_NAME = "delay_prediction_evidence.vw_delay_probability_inference_orders"


class DelayProbabilityInferenceRowNotFoundError(ValueError):
    def __init__(self, order_id: int) -> None:
        message = f"지연 확률 예측 inference row를 찾을 수 없습니다. order_id={order_id}"
        super().__init__(message)
        self.order_id = order_id


class DelayProbabilityInferenceRowDuplicatedError(ValueError):
    def __init__(self, order_id: int, row_count: int) -> None:
        message = (
            "지연 확률 예측 inference view는 order_id당 1 row만 반환해야 합니다. "
            f"order_id={order_id}, row_count={row_count}. "
            "view 내부에서 여러 production plan 정보를 주문 단위로 집계하도록 수정하세요."
        )
        super().__init__(message)
        self.order_id = order_id
        self.row_count = row_count


class DelayProbabilityRepositoryError(RuntimeError):
    pass


def _validate_sql_identifier_part(value: str) -> str:
    """
    view name과 column name을 SQL에 직접 삽입하므로
    안전한 SQL identifier만 허용합니다.
    """

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"허용되지 않는 SQL identifier입니다: {value!r}")

    return value


def _validate_qualified_view_name(view_name: str) -> str:
    parts = view_name.split(".")

    if len(parts) not in (1, 2):
        raise ValueError(
            f"view_name은 table 또는 schema.table 형태여야 합니다. 입력값: {view_name!r}"
        )

    return ".".join(_validate_sql_identifier_part(part) for part in parts)


def _build_select_clause() -> str:
    columns = get_required_inference_view_columns()

    safe_columns = [_validate_sql_identifier_part(column) for column in columns]

    return ",\n                ".join(safe_columns)


def _row_mapping_to_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    return dict(row)


class DelayProbabilityRepository:
    def __init__(
        self,
        *,
        engine: Engine = default_engine,
        view_name: str = DEFAULT_INFERENCE_VIEW_NAME,
    ) -> None:
        self.engine = engine
        self.view_name = _validate_qualified_view_name(view_name)

    def find_inference_row_by_order_id(
        self,
        *,
        order_id: int,
    ) -> dict[str, Any]:
        """
        주문 1건의 지연 확률 예측에 사용할 inference source row 1건을 조회합니다.

        view grain:
        - order_id당 1 row

        중복 row 처리:
        - 0건이면 NotFound
        - 2건 이상이면 view 설계 오류로 판단하고 DuplicatedError 발생
        """

        if order_id <= 0:
            raise ValueError(f"order_id는 1 이상이어야 합니다. 입력값: {order_id}")

        selected_columns = _build_select_clause()

        query = text(
            f"""
            SELECT
                {selected_columns}
            FROM {self.view_name}
            WHERE order_id = :order_id
            LIMIT 2
            """
        )

        try:
            with self.engine.connect() as conn:
                rows = (
                    conn.execute(
                        query,
                        {
                            "order_id": order_id,
                        },
                    )
                    .mappings()
                    .all()
                )
        except Exception as exc:
            raise DelayProbabilityRepositoryError(
                "지연 확률 예측 inference view 조회 중 오류가 발생했습니다."
            ) from exc

        if not rows:
            raise DelayProbabilityInferenceRowNotFoundError(
                order_id=order_id,
            )

        if len(rows) > 1:
            raise DelayProbabilityInferenceRowDuplicatedError(
                order_id=order_id,
                row_count=len(rows),
            )

        return _row_mapping_to_dict(rows[0])


_default_repository: DelayProbabilityRepository | None = None


def get_delay_probability_repository() -> DelayProbabilityRepository:
    global _default_repository

    if _default_repository is None:
        _default_repository = DelayProbabilityRepository()

    return _default_repository
