from argparse import Namespace
from io import StringIO

import pytest

from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.qdrant_client import QdrantCollectionCheckResult
from app.features.chat.schemas import ChatErrorCode
from scripts import check_qdrant_collection


def _build_result(is_dimension_matched: bool = True) -> QdrantCollectionCheckResult:
    return QdrantCollectionCheckResult(
        collection_name="smap_internal_documents",
        status="green",
        expected_dimension=1024,
        actual_dimension=1024 if is_dimension_matched else 384,
        is_dimension_matched=is_dimension_matched,
        points_count=12,
    )


def test_check_qdrant_collection_script_builds_settings_from_args() -> None:
    settings = check_qdrant_collection.build_settings(
        Namespace(
            qdrant_url="http://localhost:6333",
            collection="documents",
            api_key="qdrant-token",
            embedding_dimension=1024,
            timeout_seconds=3.5,
        )
    )

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "documents"
    assert settings.qdrant_api_key == "qdrant-token"
    assert settings.embedding_dimension == 1024
    assert settings.qdrant_timeout_seconds == 3.5


def test_check_qdrant_collection_script_formats_text_result() -> None:
    output = check_qdrant_collection.format_text_result(_build_result())

    assert "status=PASS" in output
    assert "collection=smap_internal_documents" in output
    assert "expectedDimension=1024" in output
    assert "actualDimension=1024" in output


def test_check_qdrant_collection_script_returns_zero_on_dimension_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_collection(settings):
        return _build_result()

    monkeypatch.setattr(check_qdrant_collection, "check_collection", fake_check_collection)
    stdout = StringIO()

    exit_code = check_qdrant_collection.main(
        ["--collection", "smap_internal_documents"],
        stdout=stdout,
    )

    assert exit_code == 0
    assert "status=PASS" in stdout.getvalue()


def test_check_qdrant_collection_script_returns_two_on_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_collection(settings):
        return _build_result(is_dimension_matched=False)

    monkeypatch.setattr(check_qdrant_collection, "check_collection", fake_check_collection)
    stdout = StringIO()

    exit_code = check_qdrant_collection.main([], stdout=stdout)

    assert exit_code == 2
    assert "status=FAIL" in stdout.getvalue()
    assert "actualDimension=384" in stdout.getvalue()


def test_check_qdrant_collection_script_prints_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_collection(settings):
        return _build_result()

    monkeypatch.setattr(check_qdrant_collection, "check_collection", fake_check_collection)
    stdout = StringIO()

    exit_code = check_qdrant_collection.main(["--json"], stdout=stdout)

    assert exit_code == 0
    assert '"checkStatus": "PASS"' in stdout.getvalue()
    assert '"status": "green"' in stdout.getvalue()
    assert '"collection_name": "smap_internal_documents"' in stdout.getvalue()


def test_check_qdrant_collection_script_returns_one_on_external_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_collection(settings):
        raise ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_QDRANT_002,
            message="Qdrant 컬렉션 조회에 실패했습니다.",
        )

    monkeypatch.setattr(check_qdrant_collection, "check_collection", fake_check_collection)
    stderr = StringIO()

    exit_code = check_qdrant_collection.main([], stderr=stderr)

    assert exit_code == 1
    assert "Qdrant 컬렉션 점검 실패" in stderr.getvalue()
    assert "CHAT_QDRANT_002" in stderr.getvalue()
