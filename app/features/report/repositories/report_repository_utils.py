# 공통 변환 함수 레포지토리
# 역할: Decimal, datetime 변환 유틸

from datetime import date, datetime
from decimal import Decimal
from typing import Any


def to_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    return value


def to_int(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, Decimal):
        return int(value)

    if isinstance(value, int):
        return value

    return int(value)


def to_iso(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    return value