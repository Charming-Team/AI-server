import pytest

from app.features.chat.query_filter_extractor import QueryFilterExtractor


@pytest.mark.parametrize(
    ("question", "expected_code"),
    [
        ("ORD-202605-001 납기 위험 알려줘", "ORD-202605-001"),
        ("line-a01 현재 병목 원인을 알려줘", "LINE-A01"),
        ("PROD-A001 생산 가능 라인을 알려줘", "PROD-A001"),
        ("mat-001 재고 현황을 알려줘", "MAT-001"),
        ("RM-AL-001 입고 예정일을 알려줘", "RM-AL-001"),
    ],
)
def test_query_filter_extractor_extracts_target_code(
    question: str,
    expected_code: str,
) -> None:
    extractor = QueryFilterExtractor()

    assert extractor.extract_target_code(question) == expected_code


def test_query_filter_extractor_returns_none_without_business_code() -> None:
    extractor = QueryFilterExtractor()

    filters = extractor.extract_filters("현재 납기 위험이 높은 주문 알려줘")

    assert filters == {
        "limit": 5,
        "fromDate": None,
        "toDate": None,
        "targetCode": None,
    }
