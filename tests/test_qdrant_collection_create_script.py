from argparse import Namespace
from io import StringIO

import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.qdrant_client import QdrantCollectionCheckResult
from app.features.chat.schemas import ChatErrorCode
from scripts import create_qdrant_collection


def _clear_qdrant_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "QDRANT_API_KEY",
        "EMBEDDING_DIMENSION",
        "QDRANT_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


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


def test_create_qdrant_collection_script_builds_settings_from_env_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_qdrant_settings_env(monkeypatch)
    env_file = tmp_path / ".env.qdrant"
    env_file.write_text(
        "\n".join(
            [
                "QDRANT_URL=http://qdrant.qdrant.svc.cluster.local:6333",
                "QDRANT_COLLECTION=env_documents",
                "QDRANT_API_KEY=env-qdrant-token",
                "EMBEDDING_DIMENSION=1024",
                "QDRANT_TIMEOUT_SECONDS=4.5",
            ]
        ),
        encoding="utf-8",
    )

    settings = create_qdrant_collection.build_settings(
        Namespace(
            qdrant_url=None,
            collection=None,
            api_key=None,
            embedding_dimension=None,
            timeout_seconds=None,
            distance="Cosine",
            json=False,
            env_file=str(env_file),
        )
    )

    assert settings.qdrant_url == "http://qdrant.qdrant.svc.cluster.local:6333"
    assert settings.qdrant_collection == "env_documents"
    assert settings.qdrant_api_key == "env-qdrant-token"
    assert settings.embedding_dimension == 1024
    assert settings.qdrant_timeout_seconds == 4.5


def test_create_qdrant_collection_script_cli_args_override_env_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_qdrant_settings_env(monkeypatch)
    env_file = tmp_path / ".env.qdrant"
    env_file.write_text(
        "\n".join(
            [
                "QDRANT_URL=http://env-qdrant:6333",
                "QDRANT_COLLECTION=env_documents",
                "QDRANT_API_KEY=env-qdrant-token",
                "EMBEDDING_DIMENSION=384",
                "QDRANT_TIMEOUT_SECONDS=9.0",
            ]
        ),
        encoding="utf-8",
    )

    settings = create_qdrant_collection.build_settings(
        Namespace(
            qdrant_url="http://cli-qdrant:6333",
            collection="cli_documents",
            api_key="cli-qdrant-token",
            embedding_dimension=1024,
            timeout_seconds=3.5,
            distance="Cosine",
            json=False,
            env_file=str(env_file),
        )
    )

    assert settings.qdrant_url == "http://cli-qdrant:6333"
    assert settings.qdrant_collection == "cli_documents"
    assert settings.qdrant_api_key == "cli-qdrant-token"
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
    assert "nextAction=" not in stdout.getvalue()


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
    assert "nextAction=EMBEDDING_DIMENSION" in stdout.getvalue()


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
    assert '"nextActions": []' in stdout.getvalue()
    assert '"error": null' in stdout.getvalue()


def test_create_qdrant_collection_script_validate_only_skips_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_ensure_collection(settings, distance):
        raise AssertionError("validate-only는 Qdrant 네트워크 생성/점검을 하면 안 됩니다.")

    monkeypatch.setattr(create_qdrant_collection, "ensure_collection", fail_ensure_collection)
    stdout = StringIO()

    exit_code = create_qdrant_collection.main(
        [
            "--validate-only",
            "--qdrant-url",
            "http://localhost:6333",
            "--collection",
            "documents",
            "--embedding-dimension",
            "1024",
            "--distance",
            "Cosine",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert "action=VALIDATED" in stdout.getvalue()
    assert "collection=documents" in stdout.getvalue()
    assert "dimensionMatched=unknown" in stdout.getvalue()
    assert "distance=Cosine" in stdout.getvalue()
    assert "networkChecked=False" in stdout.getvalue()


def test_create_qdrant_collection_script_validate_only_prints_json_without_secret() -> None:
    stdout = StringIO()

    exit_code = create_qdrant_collection.main(
        [
            "--validate-only",
            "--api-key",
            "qdrant-secret-token",
            "--json",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert '"action": "VALIDATED"' in stdout.getvalue()
    assert '"apiKeyConfigured": true' in stdout.getvalue()
    assert '"nextActions": []' in stdout.getvalue()
    assert "qdrant-secret-token" not in stdout.getvalue()


def test_create_qdrant_collection_script_validate_only_requires_qdrant_settings() -> None:
    with pytest.raises(ChatExternalServiceError) as exc_info:
        create_qdrant_collection.build_validate_only_result(
            Settings(qdrant_url=" ", qdrant_collection="documents"),
            distance="Cosine",
        )

    assert exc_info.value.code == ChatErrorCode.CHAT_QDRANT_001


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
    assert '"nextActions": [' in stdout.getvalue()
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
    assert "nextAction=Qdrant URL" in stderr.getvalue()
    assert "port-forward" in stderr.getvalue()


def test_create_qdrant_collection_script_builds_failure_actions_by_code() -> None:
    settings_error = ChatExternalServiceError(
        status_code=503,
        code=ChatErrorCode.CHAT_QDRANT_001,
        message="Qdrant 설정이 올바르지 않습니다.",
    )
    response_error = ChatExternalServiceError(
        status_code=502,
        code=ChatErrorCode.CHAT_QDRANT_003,
        message="Qdrant 응답 형식이 올바르지 않습니다.",
    )

    assert "QDRANT_URL" in create_qdrant_collection.build_collection_create_failure_actions(
        settings_error
    )[0]
    assert "응답 형식" in create_qdrant_collection.build_collection_create_failure_actions(
        response_error
    )[0]
