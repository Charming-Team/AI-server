from argparse import Namespace
from io import StringIO

import pytest

from app.core.config import Settings
from scripts import check_chat_readiness


def _ready_settings() -> Settings:
    return Settings(
        chat_answer_internal_token="answer-token",
        chat_recommendation_internal_token="recommendation-token",
        document_index_internal_token="document-token",
        evidence_lookup_enabled=True,
        evidence_lookup_internal_token="evidence-token",
        llm_enabled=True,
    )


def _rdb_ready_settings() -> Settings:
    return Settings(
        chat_answer_internal_token="answer-token",
        chat_recommendation_internal_token="recommendation-token",
        document_index_internal_token="document-token",
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
        llm_enabled=True,
    )


def _vector_ready_settings() -> Settings:
    return Settings(
        chat_answer_internal_token="answer-token",
        chat_recommendation_internal_token="recommendation-token",
        document_index_internal_token="document-token",
        qdrant_search_enabled=True,
        qdrant_url="http://qdrant.local:6333",
        qdrant_collection="smap_internal_documents",
        embedding_enabled=True,
        embedding_base_url="http://embedding.local",
        embedding_path="/embed",
        embedding_model="BAAI/bge-m3",
        llm_enabled=True,
    )


def test_check_chat_readiness_script_builds_settings_from_env_file(
    tmp_path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APP_NAME=S-MAP Test AI Server",
                "CHAT_ANSWER_INTERNAL_TOKEN=answer-token",
                "CHAT_RECOMMENDATION_INTERNAL_TOKEN=recommendation-token",
                "DOCUMENT_INDEX_INTERNAL_TOKEN=document-token",
                "EVIDENCE_LOOKUP_ENABLED=true",
                "EVIDENCE_LOOKUP_INTERNAL_TOKEN=evidence-token",
                "LLM_ENABLED=true",
            ]
        ),
        encoding="utf-8",
    )

    settings = check_chat_readiness.build_settings(
        Namespace(env_file=str(env_file), json=False)
    )

    assert settings.app_name == "S-MAP Test AI Server"
    assert settings.evidence_lookup_enabled is True
    assert settings.llm_enabled is True


def test_check_chat_readiness_script_builds_ready_result() -> None:
    result = check_chat_readiness.build_readiness_result(_ready_settings())

    components = {component["name"]: component for component in result["components"]}
    assert result["status"] == "ready"
    assert result["requirementFailures"] == []
    assert components["chatGroundingPipeline"]["configured"] is True
    assert components["answerGenerationPipeline"]["configured"] is True


def test_check_chat_readiness_script_builds_required_components() -> None:
    args = Namespace(
        require_rdb_evidence=True,
        require_vector_search=True,
        require_document_index=True,
        require_llm_generation=True,
    )

    assert check_chat_readiness.build_required_components(args) == [
        "rdbEvidence",
        "qdrantSearch",
        "ragSearchPipeline",
        "documentIndexPipeline",
        "llm",
    ]


def test_check_chat_readiness_script_formats_text_result() -> None:
    result = check_chat_readiness.build_readiness_result(Settings())

    output = check_chat_readiness.format_text_result(result)

    assert "status=not_ready" in output
    assert "chatAnswerInternalToken: enabled=True configured=False" in output
    assert "code=CHAT_SECURITY_003" in output


def test_check_chat_readiness_script_marks_required_rdb_evidence_failure() -> None:
    result = check_chat_readiness.build_readiness_result(
        _ready_settings(),
        required_components=["rdbEvidence"],
    )

    assert result["status"] == "not_ready"
    assert result["requirementFailures"] == [
        {
            "name": "rdbEvidence",
            "code": "CHAT_EVIDENCE_004",
            "reason": "비활성화되어 있습니다.",
        }
    ]


def test_check_chat_readiness_script_accepts_required_rdb_evidence() -> None:
    result = check_chat_readiness.build_readiness_result(
        _rdb_ready_settings(),
        required_components=["rdbEvidence"],
    )

    assert result["status"] == "ready"
    assert result["requirementFailures"] == []


def test_check_chat_readiness_script_marks_vector_search_requirement_failures() -> None:
    result = check_chat_readiness.build_readiness_result(
        _rdb_ready_settings(),
        required_components=["qdrantSearch", "ragSearchPipeline"],
    )

    assert result["status"] == "not_ready"
    assert result["requirementFailures"] == [
        {
            "name": "qdrantSearch",
            "code": "CHAT_QDRANT_001",
            "reason": "비활성화되어 있습니다.",
        },
        {
            "name": "ragSearchPipeline",
            "code": "CHAT_EMBEDDING_001",
            "reason": "Qdrant 검색이 비활성화되어 있습니다.",
        },
    ]


def test_check_chat_readiness_script_accepts_required_vector_search() -> None:
    result = check_chat_readiness.build_readiness_result(
        _vector_ready_settings(),
        required_components=["qdrantSearch", "ragSearchPipeline"],
    )

    assert result["status"] == "ready"
    assert result["requirementFailures"] == []


def test_check_chat_readiness_script_marks_required_llm_generation_failure() -> None:
    result = check_chat_readiness.build_readiness_result(
        Settings(
            chat_answer_internal_token="answer-token",
            chat_recommendation_internal_token="recommendation-token",
            document_index_internal_token="document-token",
            evidence_lookup_enabled=True,
            evidence_lookup_internal_token="evidence-token",
            llm_enabled=False,
        ),
        required_components=["llm"],
    )

    assert result["status"] == "not_ready"
    assert result["requirementFailures"] == [
        {
            "name": "llm",
            "code": "CHAT_LLM_001",
            "reason": "비활성화되어 있습니다.",
        }
    ]


def test_check_chat_readiness_script_returns_zero_on_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_chat_readiness,
        "build_settings",
        lambda args: _ready_settings(),
    )
    stdout = StringIO()

    exit_code = check_chat_readiness.main([], stdout=stdout)

    assert exit_code == 0
    assert "status=ready" in stdout.getvalue()


def test_check_chat_readiness_script_returns_two_on_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_chat_readiness,
        "build_settings",
        lambda args: Settings(),
    )
    stdout = StringIO()

    exit_code = check_chat_readiness.main([], stdout=stdout)

    assert exit_code == 2
    assert "status=not_ready" in stdout.getvalue()


def test_check_chat_readiness_script_returns_two_on_requirement_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_chat_readiness,
        "build_settings",
        lambda args: _ready_settings(),
    )
    stdout = StringIO()

    exit_code = check_chat_readiness.main(
        ["--require-rdb-evidence"],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 2
    assert "status=not_ready" in output
    assert "requirementFailure: name=rdbEvidence" in output


def test_check_chat_readiness_script_prints_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_chat_readiness,
        "build_settings",
        lambda args: _ready_settings(),
    )
    stdout = StringIO()

    exit_code = check_chat_readiness.main(["--json"], stdout=stdout)

    assert exit_code == 0
    assert '"status": "ready"' in stdout.getvalue()
    assert '"name": "chatGroundingPipeline"' in stdout.getvalue()


@pytest.mark.parametrize("argv", [[], ["--json"]])
def test_check_chat_readiness_script_does_not_expose_secret_values(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_chat_readiness,
        "build_settings",
        lambda args: _ready_settings(),
    )
    stdout = StringIO()

    exit_code = check_chat_readiness.main(argv, stdout=stdout)

    output = stdout.getvalue()
    assert exit_code == 0
    assert "answer-token" not in output
    assert "recommendation-token" not in output
    assert "document-token" not in output
    assert "evidence-token" not in output
    assert "reader:secret" not in output


def test_check_chat_readiness_script_returns_one_on_settings_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_error(args):
        raise ValueError("invalid config")

    monkeypatch.setattr(check_chat_readiness, "build_settings", raise_error)
    stderr = StringIO()

    exit_code = check_chat_readiness.main([], stderr=stderr)

    assert exit_code == 1
    assert "챗봇 readiness 점검 실패" in stderr.getvalue()
