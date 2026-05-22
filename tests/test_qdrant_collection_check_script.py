from argparse import Namespace
from io import StringIO

import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.qdrant_client import QdrantCollectionCheckResult
from app.features.chat.schemas import ChatErrorCode
from scripts import check_qdrant_collection


def _clear_qdrant_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "QDRANT_API_KEY",
        "EMBEDDING_DIMENSION",
        "QDRANT_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


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


def test_check_qdrant_collection_script_builds_settings_from_env_file(
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

    settings = check_qdrant_collection.build_settings(
        Namespace(
            qdrant_url=None,
            collection=None,
            api_key=None,
            embedding_dimension=None,
            timeout_seconds=None,
            env_file=str(env_file),
        )
    )

    assert settings.qdrant_url == "http://qdrant.qdrant.svc.cluster.local:6333"
    assert settings.qdrant_collection == "env_documents"
    assert settings.qdrant_api_key == "env-qdrant-token"
    assert settings.embedding_dimension == 1024
    assert settings.qdrant_timeout_seconds == 4.5


def test_check_qdrant_collection_script_cli_args_override_env_file(
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

    settings = check_qdrant_collection.build_settings(
        Namespace(
            qdrant_url="http://cli-qdrant:6333",
            collection="cli_documents",
            api_key="cli-qdrant-token",
            embedding_dimension=1024,
            timeout_seconds=3.5,
            env_file=str(env_file),
        )
    )

    assert settings.qdrant_url == "http://cli-qdrant:6333"
    assert settings.qdrant_collection == "cli_documents"
    assert settings.qdrant_api_key == "cli-qdrant-token"
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
    assert "code=CHAT_QDRANT_004" in stdout.getvalue()
    assert "nextAction=EMBEDDING_DIMENSION" in stdout.getvalue()


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
    assert '"error": null' in stdout.getvalue()


def test_check_qdrant_collection_script_validate_only_skips_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_check_collection(settings):
        raise AssertionError("validate-only는 Qdrant 네트워크 조회를 하면 안 됩니다.")

    monkeypatch.setattr(check_qdrant_collection, "check_collection", fail_check_collection)
    stdout = StringIO()

    exit_code = check_qdrant_collection.main(
        [
            "--validate-only",
            "--qdrant-url",
            "http://localhost:6333",
            "--collection",
            "documents",
            "--embedding-dimension",
            "1024",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert "status=VALIDATED" in stdout.getvalue()
    assert "mode=validateOnly" in stdout.getvalue()
    assert "collection=documents" in stdout.getvalue()
    assert "networkChecked=False" in stdout.getvalue()


def test_check_qdrant_collection_script_validate_only_prints_json_without_secret() -> None:
    stdout = StringIO()

    exit_code = check_qdrant_collection.main(
        [
            "--validate-only",
            "--api-key",
            "qdrant-secret-token",
            "--json",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert '"checkStatus": "VALIDATED"' in stdout.getvalue()
    assert '"apiKeyConfigured": true' in stdout.getvalue()
    assert "qdrant-secret-token" not in stdout.getvalue()


def test_check_qdrant_collection_script_validate_only_requires_qdrant_settings() -> None:
    with pytest.raises(ChatExternalServiceError) as exc_info:
        check_qdrant_collection.build_validate_only_result(
            Settings(qdrant_url=" ", qdrant_collection="documents")
        )

    assert exc_info.value.code == ChatErrorCode.CHAT_QDRANT_001


@pytest.mark.parametrize(
    "argv",
    [
        ["--api-key", "qdrant-secret-token"],
        ["--api-key", "qdrant-secret-token", "--json"],
    ],
)
def test_check_qdrant_collection_script_does_not_expose_api_key(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_collection(settings):
        assert settings.qdrant_api_key == "qdrant-secret-token"
        return _build_result()

    monkeypatch.setattr(check_qdrant_collection, "check_collection", fake_check_collection)
    stdout = StringIO()

    exit_code = check_qdrant_collection.main(argv, stdout=stdout)

    assert exit_code == 0
    assert "qdrant-secret-token" not in stdout.getvalue()


def test_check_qdrant_collection_script_prints_json_error_on_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_collection(settings):
        return _build_result(is_dimension_matched=False)

    monkeypatch.setattr(check_qdrant_collection, "check_collection", fake_check_collection)
    stdout = StringIO()

    exit_code = check_qdrant_collection.main(["--json"], stdout=stdout)

    assert exit_code == 2
    assert '"checkStatus": "FAIL"' in stdout.getvalue()
    assert '"code": "CHAT_QDRANT_004"' in stdout.getvalue()
    assert '"nextActions": [' in stdout.getvalue()
    assert "FastAPI 임베딩 설정과 일치하지 않습니다" in stdout.getvalue()


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
    assert "nextAction=QDRANT_URL" in stderr.getvalue()
    assert "kubectl port-forward" in stderr.getvalue()


def test_check_qdrant_collection_script_guides_collection_creation_on_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_collection(settings):
        raise ChatExternalServiceError(
            status_code=404,
            code=ChatErrorCode.CHAT_QDRANT_002,
            message="Qdrant 컬렉션 조회에 실패했습니다.",
        )

    monkeypatch.setattr(check_qdrant_collection, "check_collection", fake_check_collection)
    stderr = StringIO()

    exit_code = check_qdrant_collection.main([], stderr=stderr)

    assert exit_code == 1
    assert "nextAction=QDRANT_COLLECTION" in stderr.getvalue()
    assert "scripts.create_qdrant_collection" in stderr.getvalue()


def test_check_qdrant_collection_script_builds_actions_by_error_code() -> None:
    assert check_qdrant_collection.build_collection_failure_actions(
        ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_QDRANT_001,
            message="missing",
        )
    ) == ["QDRANT_URL과 QDRANT_COLLECTION 설정을 확인하세요."]

    assert check_qdrant_collection.build_collection_failure_actions(
        ChatExternalServiceError(
            status_code=502,
            code=ChatErrorCode.CHAT_QDRANT_003,
            message="invalid",
        )
    ) == [
        "Qdrant 호환 API를 호출 중인지 확인하세요.",
        "프록시나 Ingress가 Qdrant JSON 응답을 HTML/오류 페이지로 바꾸지 않는지 확인하세요.",
    ]
