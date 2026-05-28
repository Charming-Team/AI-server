from datetime import datetime

import pytest

from app.features.chat.query_filter_extractor import QueryFilterExtractor


@pytest.mark.parametrize(
    ("question", "expected_type", "expected_code"),
    [
        ("ORD-202605-001 납기 위험 알려줘", "ORDER", "ORD-202605-001"),
        ("line-a01 현재 병목 원인을 알려줘", "LINE", "LINE-A01"),
        ("LINE-PE-01 현재 병목 원인을 알려줘", "LINE", "LINE-PE-01"),
        ("line-abs-01 전환 기준 알려줘", "LINE", "LINE-ABS-01"),
        ("PROD-A001 생산 가능 라인을 알려줘", "PRODUCT", "PROD-A001"),
        ("mat-001 재고 현황을 알려줘", "MATERIAL", "MAT-001"),
        ("RM-AL-001 입고 예정일을 알려줘", "MATERIAL", "RM-AL-001"),
    ],
)
def test_query_filter_extractor_extracts_target(
    question: str,
    expected_type: str,
    expected_code: str,
) -> None:
    extractor = QueryFilterExtractor()

    assert extractor.extract_target_type(question) == expected_type
    assert extractor.extract_target_code(question) == expected_code


def test_query_filter_extractor_returns_none_without_business_code() -> None:
    extractor = QueryFilterExtractor()

    filters = extractor.extract_filters("현재 납기 위험이 높은 주문 알려줘")

    assert filters == {
        "limit": 5,
        "fromDate": None,
        "toDate": None,
        "targetType": None,
        "targetCode": None,
    }


def test_query_filter_extractor_expands_limit_for_count_questions() -> None:
    extractor = QueryFilterExtractor()

    filters = extractor.extract_filters("우리 공정 라인은 몇개 있어?")

    assert filters["limit"] == 50


@pytest.mark.parametrize(
    "question",
    [
        "현재 가동 중인 라인은 뭐야?",
        "생산 라인 구성 알려줘",
        "전체 라인 상태 알려줘",
    ],
)
def test_query_filter_extractor_expands_limit_for_line_overview_questions(
    question: str,
) -> None:
    extractor = QueryFilterExtractor()

    filters = extractor.extract_filters(question)

    assert filters["limit"] == 50


def test_query_filter_extractor_keeps_default_limit_for_line_bottleneck_question() -> None:
    extractor = QueryFilterExtractor()

    filters = extractor.extract_filters("라인 병목 현황 알려줘")

    assert filters["limit"] == 5


def test_query_filter_extractor_keeps_default_limit_for_all_line_bottleneck_question() -> None:
    extractor = QueryFilterExtractor()

    filters = extractor.extract_filters("전체 라인 병목 현황 알려줘")

    assert filters["limit"] == 5


@pytest.mark.parametrize(
    ("question", "expected_from_date", "expected_to_date"),
    [
        ("오늘 납기 위험 알려줘", "2026-05-12", "2026-05-12"),
        ("내일 생산계획 알려줘", "2026-05-13", "2026-05-13"),
        ("어제 라인 병목 알려줘", "2026-05-11", "2026-05-11"),
        ("이번 주 자재 부족 알려줘", "2026-05-11", "2026-05-17"),
        ("다음주 작업 우선순위 알려줘", "2026-05-18", "2026-05-24"),
        ("이번달 보고서 알려줘", "2026-05-01", "2026-05-31"),
        ("다음 달 납기 위험 알려줘", "2026-06-01", "2026-06-30"),
    ],
)
def test_query_filter_extractor_extracts_relative_date_range(
    question: str,
    expected_from_date: str,
    expected_to_date: str,
) -> None:
    extractor = QueryFilterExtractor()
    reference_datetime = datetime.fromisoformat("2026-05-12T10:30:00+09:00")

    filters = extractor.extract_filters(question, reference_datetime)

    assert filters["fromDate"] == expected_from_date
    assert filters["toDate"] == expected_to_date


def test_query_filter_extractor_extracts_explicit_date_range() -> None:
    extractor = QueryFilterExtractor()

    filters = extractor.extract_filters(
        "2026-05-01부터 2026.05.31까지 납기 위험 알려줘"
    )

    assert filters["fromDate"] == "2026-05-01"
    assert filters["toDate"] == "2026-05-31"


def test_query_filter_extractor_extracts_single_explicit_date() -> None:
    extractor = QueryFilterExtractor()

    filters = extractor.extract_filters("2026/05/12 LINE-A01 병목 알려줘")

    assert filters["fromDate"] == "2026-05-12"
    assert filters["toDate"] == "2026-05-12"
    assert filters["targetType"] == "LINE"
    assert filters["targetCode"] == "LINE-A01"
