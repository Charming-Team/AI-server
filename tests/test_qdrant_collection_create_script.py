from argparse import Namespace
from io import StringIO

import pytest

from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.qdrant_client import QdrantCollectionCheckResult
from app.features.chat.schemas import ChatErrorCode
from scripts import create_qdrant_collection


def _build_check_result(
    is_dimension_matched: bool = True,
    actual_dimension: int = 1024,
) -> QdrantCollectionCheckResult:
    return QdrantCollectionCheckResult(
        collection_name="smap_internal_documents",
        status="green",
        expected_dimension=1024,
        actual_dimension=actual_dimension,
        is_dimension_matched=is_dimension_matched,
        points_count=0,
    )


def test_create_qdrant_collection_script_builds_settings_from_args() -> None:
    settings = create_qdrant_collection.build_settings(
        Namespace(
            qdrant_url="http://localhost:6333",
            collection="smap_internal_documents",
            api_key="qdrant-token",
            embedding_dimension=1024,
            timeout_seconds=3.5,
            distance="Cosine",
            json=False,
        )
    )

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "smap_internal_documents"
    assert settings.qdrant_api_key == "qdrant-token"
    assert settings.embedding_dimension == 1024
    assert settings.qdrant_timeout_seconds == 3.5


def test_create_qdrant_collection_script_keeps_existing_matching_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_collection(settings):
        return _build_check_result()

    async def fake_create_collection(settings, distance):
        raise AssertionError("dimension이 맞는 기존 컬렉션은 다시 생성하면 안 됩니다.")

    monkeypatch.setattr(create_qdrant_collection, "check_collection", fake_check_collection)
    monkeypatch.setattr(create_qdrant_collection, "create_collection", fake_create_collection)
    stdout = StringIO()

    exit_code = create_qdrant_collection.main([], stdout=stdout)

    assert exit_code == 0
    assert "action=EXISTS" in stdout.getvalue()
    assert "dimensionMatched=True" in stdout.getvalue()


def test_create_qdrant_collection_script_creates_missing_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    async def fake_check_collection(settings):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ChatExternalServiceError(
                status_code=404,
                code=ChatErrorCode.CHAT_QDRANT_002,
                message="Qdrant 컬렉션 조회에 실패했습니다.",
            )
        return _build_check_result()

    async def fake_create_collection(settings, distance):
        return {"result": True, "status": "ok", "distance": distance}

    monkeypatch.setattr(create_qdrant_collection, "check_collection", fake_check_collection)
    monkeypatch.setattr(create_qdrant_collection, "create_collection", fake_create_collection)
    stdout = StringIO()

    exit_code = create_qdrant_collection.main(
        ["--distance", "Cosine"],
        stdout=stdout,
    )

    assert exit_code == 0
    assert "action=CREATED" in stdout.getvalue()
    assert call_count == 2


def test_create_qdrant_collection_script_returns_two_on_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_collection(settings):
        return _build_check_result(is_dimension_matched=False, actual_dimension=1536)

    monkeypatch.setattr(create_qdrant_collection, "check_collection", fake_check_collection)
    stdout = StringIO()

    exit_code = create_qdrant_collection.main([], stdout=stdout)

    assert exit_code == 2
    assert "action=MISMATCH" in stdout.getvalue()
    assert "actualDimension=1536" in stdout.getvalue()
    assert "code=CHAT_QDRANT_004" in stdout.getvalue()


def test_create_qdrant_collection_script_prints_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_collection(settings):
        return _build_check_result()

    monkeypatch.setattr(create_qdrant_collection, "check_collection", fake_check_collection)
    stdout = StringIO()

    exit_code = create_qdrant_collection.main(["--json"], stdout=stdout)

    assert exit_code == 0
    assert '"action": "EXISTS"' in stdout.getvalue()
    assert '"collection_name": "smap_internal_documents"' in stdout.getvalue()
    assert '"error": null' in stdout.getvalue()


@pytest.mark.parametrize(
    "argv",
    [
        ["--api-key", "qdrant-secret-token"],
        ["--api-key", "qdrant-secret-token", "--json"],
    ],
)
def test_create_qdrant_collection_script_does_not_expose_api_key(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_collection(settings):
        assert settings.qdrant_api_key == "qdrant-secret-token"
        return _build_check_result()

    monkeypatch.setattr(create_qdrant_collection, "check_collection", fake_check_collection)
    stdout = StringIO()

    exit_code = create_qdrant_collection.main(argv, stdout=stdout)

    assert exit_code == 0
    assert "qdrant-secret-token" not in stdout.getvalue()


def test_create_qdrant_collection_script_prints_json_error_on_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_collection(settings):
        return _build_check_result(is_dimension_matched=False, actual_dimension=1536)

    monkeypatch.setattr(create_qdrant_collection, "check_collection", fake_check_collection)
    stdout = StringIO()

    exit_code = create_qdrant_collection.main(["--json"], stdout=stdout)

    assert exit_code == 2
    assert '"action": "MISMATCH"' in stdout.getvalue()
    assert '"code": "CHAT_QDRANT_004"' in stdout.getvalue()
    assert "FastAPI 임베딩 설정과 일치하지 않습니다" in stdout.getvalue()


def test_create_qdrant_collection_script_returns_one_on_external_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_collection(settings):
        raise ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_QDRANT_002,
            message="Qdrant 컬렉션 조회에 실패했습니다.",
        )

    monkeypatch.setattr(create_qdrant_collection, "check_collection", fake_check_collection)
    stderr = StringIO()

    exit_code = create_qdrant_collection.main([], stderr=stderr)

    assert exit_code == 1
    assert "Qdrant 컬렉션 생성/점검 실패" in stderr.getvalue()
    assert "CHAT_QDRANT_002" in stderr.getvalue()
