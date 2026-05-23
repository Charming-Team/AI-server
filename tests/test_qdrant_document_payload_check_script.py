from argparse import Namespace
from io import StringIO
from typing import Any

import anyio
import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError, ChatServiceError
from app.features.chat.schemas import ChatErrorCode
from scripts import check_qdrant_document_payloads


def _build_args(**overrides: Any) -> Namespace:
    values = {
        "qdrant_url": "http://qdrant.local:6333",
        "collection": "smap_internal_documents",
        "api_key": "qdrant-token",
        "env_file": None,
        "limit": 20,
        "min_points": 0,
        "timeout_seconds": None,
        "json": False,
        "validate_only": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _valid_point(**payload_overrides: Any) -> dict:
    payload = {
        "documentId": "report-202605",
        "documentType": "REPORT",
        "title": "2026년 5월 생산 리스크 보고서",
        "chunkText": "자재 부족과 라인 병목이 주요 리스크입니다.",
        "chunkId": "chunk-0001",
        "url": "/reports/20",
        "allowedRoles": ["EXECUTIVE", "MANUFACTURING_MANAGER"],
        "intentTags": ["REPORT_LOOKUP"],
    }
    payload.update(payload_overrides)
    return {"id": "point-1", "payload": payload}


class FakeQdrantClient:
    def __init__(self, points: list[dict]) -> None:
        self.points = points
        self.limit: int | None = None

    async def scroll_points(self, limit: int = 20) -> list[dict]:
        self.limit = limit
        return self.points


def test_check_qdrant_document_payloads_builds_settings_from_cli() -> None:
    settings = check_qdrant_document_payloads.build_settings(
        _build_args(
            qdrant_url="http://qdrant.qdrant.svc:6333",
            collection="documents",
            api_key="secret-token",
            timeout_seconds=3.5,
        )
    )

    assert settings.qdrant_url == "http://qdrant.qdrant.svc:6333"
    assert settings.qdrant_collection == "documents"
    assert settings.qdrant_api_key == "secret-token"
    assert settings.qdrant_timeout_seconds == 3.5


def test_check_qdrant_document_payloads_validate_only_result() -> None:
    result = check_qdrant_document_payloads.build_validate_only_result(
        Settings(qdrant_url="http://qdrant.local", qdrant_collection="docs"),
        limit=10,
        min_points=1,
    )

    assert result["checkStatus"] == "VALIDATED"
    assert result["mode"] == "VALIDATE_ONLY"
    assert result["collectionName"] == "docs"
    assert result["qdrantUrlConfigured"] is True
    assert result["networkChecked"] is False
    assert result["limit"] == 10
    assert result["minPoints"] == 1


def test_check_qdrant_document_payloads_passes_valid_points() -> None:
    client = FakeQdrantClient([_valid_point()])

    async def run() -> dict:
        return await check_qdrant_document_payloads.check_qdrant_document_payloads(
            Settings(qdrant_url="http://qdrant.local", qdrant_collection="docs"),
            limit=5,
            min_points=1,
            client=client,
        )

    result = anyio.run(run)

    assert client.limit == 5
    assert result["checkStatus"] == "PASS"
    assert result["pointCount"] == 1
    assert result["invalidCount"] == 0
    assert result["documentTypes"] == ["REPORT"]
    assert result["intentTags"] == ["REPORT_LOOKUP"]
    assert result["allowedRoles"] == ["EXECUTIVE", "MANUFACTURING_MANAGER"]


@pytest.mark.parametrize(
    ("point", "expected_error"),
    [
        ({"id": "point-1"}, "payload must be object"),
        (_valid_point(documentType="UNKNOWN"), "documentType must be REPORT"),
        (_valid_point(allowedRoles=[]), "allowedRoles is required"),
        (_valid_point(allowedRoles=["ADMIN"]), "unsupported role"),
        (_valid_point(intentTags=[]), "intentTags is required"),
        (_valid_point(intentTags=["UNKNOWN"]), "unsupported intent"),
        (_valid_point(url="https://external.example/report"), "internal relative path"),
        (
            _valid_point(url=None, referenceType=None, referenceId=None),
            "url or reference metadata is required",
        ),
        (
            _valid_point(url=None, referenceType=" ", referenceId=20),
            "url or reference metadata is required",
        ),
        (
            _valid_point(url=None, referenceType="REPORT", referenceId=None),
            "url or reference metadata is required",
        ),
        (
            _valid_point(url=None, referenceType="REPORT", referenceId=0),
            "url or reference metadata is required",
        ),
        (
            _valid_point(url=None, referenceType="REPORT", referenceId=-1),
            "url or reference metadata is required",
        ),
        (
            _valid_point(
                allowedRoles=["OPERATOR"],
                chunkText="계약 금액과 패널티가 포함된 보고서입니다.",
            ),
            "restricted business terms",
        ),
    ],
)
def test_validate_points_returns_payload_contract_errors(
    point: dict,
    expected_error: str,
) -> None:
    invalid_points = check_qdrant_document_payloads.validate_points([point])

    assert len(invalid_points) == 1
    assert any(expected_error in error for error in invalid_points[0]["errors"])


def test_validate_points_accepts_reference_metadata_without_url() -> None:
    point = _valid_point(url=None, referenceType="REPORT", referenceId=20)

    invalid_points = check_qdrant_document_payloads.validate_points([point])

    assert invalid_points == []


def test_validate_points_allows_restricted_terms_when_operator_is_not_allowed() -> None:
    point = _valid_point(
        allowedRoles=["EXECUTIVE", "MANUFACTURING_MANAGER"],
        chunkText="매출 구조와 비용 정보가 포함된 경영 보고서입니다.",
    )

    invalid_points = check_qdrant_document_payloads.validate_points([point])

    assert invalid_points == []


def test_check_qdrant_document_payloads_fails_below_minimum_points() -> None:
    async def run() -> None:
        await check_qdrant_document_payloads.check_qdrant_document_payloads(
            Settings(qdrant_url="http://qdrant.local", qdrant_collection="docs"),
            min_points=1,
            client=FakeQdrantClient([]),
        )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_QDRANT_004"
    assert "expected>=1, actual=0" in exc_info.value.message


def test_check_qdrant_document_payloads_fails_invalid_payloads() -> None:
    async def run() -> None:
        await check_qdrant_document_payloads.check_qdrant_document_payloads(
            Settings(qdrant_url="http://qdrant.local", qdrant_collection="docs"),
            client=FakeQdrantClient([_valid_point(intentTags=[])]),
        )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_QDRANT_003"
    assert "invalidCount=1" in exc_info.value.message


def test_check_qdrant_document_payloads_main_does_not_expose_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_qdrant_document_payloads(*args: Any, **kwargs: Any) -> dict:
        return {
            "checkStatus": "PASS",
            "mode": "NETWORK",
            "collectionName": "docs",
            "networkChecked": True,
            "limit": 20,
            "minPoints": 0,
            "pointCount": 1,
            "invalidCount": 0,
            "documentTypes": ["REPORT"],
            "intentTags": ["REPORT_LOOKUP"],
            "allowedRoles": ["EXECUTIVE"],
        }

    monkeypatch.setattr(
        check_qdrant_document_payloads,
        "check_qdrant_document_payloads",
        fake_check_qdrant_document_payloads,
    )
    stdout = StringIO()

    exit_code = check_qdrant_document_payloads.main(
        ["--api-key", "secret-qdrant-token"],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "status=PASS" in output
    assert "secret-qdrant-token" not in output


def test_check_qdrant_document_payloads_main_returns_one_for_missing_settings() -> None:
    stderr = StringIO()

    exit_code = check_qdrant_document_payloads.main(
        ["--qdrant-url", " "],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "Qdrant 문서 payload 점검 실패" in stderr.getvalue()
    assert "code=CHAT_QDRANT_001" in stderr.getvalue()
    assert "nextAction=QDRANT_URL" in stderr.getvalue()


def test_check_qdrant_document_payloads_main_guides_min_points_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_qdrant_document_payloads(*args: Any, **kwargs: Any) -> dict:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_QDRANT_004,
            message="Qdrant 문서 point 개수가 기준보다 적습니다. expected>=1, actual=0",
        )

    monkeypatch.setattr(
        check_qdrant_document_payloads,
        "check_qdrant_document_payloads",
        fake_check_qdrant_document_payloads,
    )
    stderr = StringIO()

    exit_code = check_qdrant_document_payloads.main(
        ["--min-points", "1"],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "code=CHAT_QDRANT_004" in stderr.getvalue()
    assert "nextAction=보고서 또는 회사정보 문서" in stderr.getvalue()
    assert "--min-points 0" in stderr.getvalue()


def test_check_qdrant_document_payloads_main_guides_contract_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_qdrant_document_payloads(*args: Any, **kwargs: Any) -> dict:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_QDRANT_003,
            message="Qdrant 문서 payload 계약을 만족하지 않는 point가 있습니다. invalidCount=1",
        )

    monkeypatch.setattr(
        check_qdrant_document_payloads,
        "check_qdrant_document_payloads",
        fake_check_qdrant_document_payloads,
    )
    stderr = StringIO()

    exit_code = check_qdrant_document_payloads.main([], stderr=stderr)

    assert exit_code == 1
    assert "code=CHAT_QDRANT_003" in stderr.getvalue()
    assert "nextAction=Qdrant payload의 documentId" in stderr.getvalue()
    assert "OPERATOR 허용 문서" in stderr.getvalue()


def test_check_qdrant_document_payloads_builds_actions_by_error_code() -> None:
    assert check_qdrant_document_payloads.build_payload_failure_actions(
        ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_QDRANT_002,
            message="Qdrant 문서 payload 조회에 실패했습니다.",
        )
    ) == [
        "Qdrant URL, collection 이름, port-forward 상태를 확인하세요.",
        "컬렉션이 없다면 scripts.create_qdrant_collection으로 먼저 생성하세요.",
    ]

    assert check_qdrant_document_payloads.build_payload_failure_actions(
        ChatServiceError(
            status_code=502,
            code=ChatErrorCode.CHAT_QDRANT_003,
            message="Qdrant 응답 형식이 올바르지 않습니다.",
        )
    ) == ["Qdrant scroll API 응답 형식이 예상 JSON 구조인지 확인하세요."]
