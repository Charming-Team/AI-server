from io import StringIO
from pathlib import Path

from scripts import check_k8s_runtime_env


def _valid_env_values() -> dict[str, str]:
    return {
        "ENVIRONMENT": "prod",
        "API_V1_PREFIX": "/api/v1",
        "CHAT_ANSWER_INTERNAL_TOKEN": "answer-token",
        "CHAT_RECOMMENDATION_INTERNAL_TOKEN": "recommendation-token",
        "DOCUMENT_INDEX_INTERNAL_TOKEN": "document-token",
        "EVIDENCE_LOOKUP_ENABLED": "false",
        "RDB_EVIDENCE_ENABLED": "true",
        "RDB_EVIDENCE_DSN": "postgresql://reader:secret@postgres.local:5432/smap",
        "QDRANT_SEARCH_ENABLED": "true",
        "QDRANT_URL": "http://qdrant.qdrant.svc.cluster.local:6333",
        "QDRANT_COLLECTION": "smap_internal_documents",
        "EMBEDDING_ENABLED": "true",
        "EMBEDDING_BASE_URL": (
            "http://embedding-service.skala3-finalproj-class3-team12.svc.cluster.local:8002"
        ),
        "EMBEDDING_PATH": "/embed",
        "EMBEDDING_MODEL": "BAAI/bge-m3",
        "EMBEDDING_DIMENSION": "1024",
        "LLM_ENABLED": "true",
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_API_KEY": "openai-secret-token",
        "LLM_MODEL": "gpt-5-nano",
        "LLM_ALLOWED_MODELS": "gpt-5-nano",
        "LLM_REASONING_EFFORT": "minimal",
        "LLM_MAX_TOKENS": "1024",
        "LLM_RESPONSE_CACHE_ENABLED": "true",
        "LLM_RESPONSE_CACHE_TTL_SECONDS": "60.0",
        "LLM_RESPONSE_CACHE_MAX_ENTRIES": "128",
        "ANSWER_MAX_CHARS": "900",
        "PROMPT_MAX_EVIDENCE_ITEMS": "3",
        "PROMPT_MAX_DOCUMENT_SOURCES": "2",
        "PROMPT_MAX_SUMMARY_CHARS": "280",
        "PROMPT_MAX_TOTAL_CHARS": "3000",
    }


def test_check_k8s_runtime_env_accepts_runtime_values() -> None:
    result = check_k8s_runtime_env.check_k8s_runtime_env(_valid_env_values())

    assert result["checkStatus"] == "PASS"
    assert result["failureCount"] == 0


def test_check_k8s_runtime_env_accepts_standard_openai_api_key_name() -> None:
    values = _valid_env_values()
    values.pop("LLM_API_KEY")
    values["OPENAI_API_KEY"] = "openai-secret-token"

    result = check_k8s_runtime_env.check_k8s_runtime_env(values)

    assert result["checkStatus"] == "PASS"


def test_check_k8s_runtime_env_allows_placeholders_for_example_file() -> None:
    values = _valid_env_values()
    values["CHAT_ANSWER_INTERNAL_TOKEN"] = "__SET_BY_SECRET__"
    values["CHAT_RECOMMENDATION_INTERNAL_TOKEN"] = "__SET_BY_SECRET__"
    values["DOCUMENT_INDEX_INTERNAL_TOKEN"] = "__SET_BY_SECRET__"
    values["LLM_API_KEY"] = "__SET_BY_SECRET__"
    values["LLM_MODEL"] = "__SET_OPENAI_MODEL__"
    values["LLM_ALLOWED_MODELS"] = "__SET_OPENAI_ALLOWED_MODELS__"
    values["RDB_EVIDENCE_DSN"] = "__SET_BY_SECRET__"

    result = check_k8s_runtime_env.check_k8s_runtime_env(
        values,
        allow_placeholders=True,
    )

    assert result["checkStatus"] == "PASS"


def test_check_k8s_runtime_env_rejects_placeholders_for_actual_runtime() -> None:
    values = _valid_env_values()
    values["CHAT_ANSWER_INTERNAL_TOKEN"] = "__SET_BY_SECRET__"

    result = check_k8s_runtime_env.check_k8s_runtime_env(values)

    assert result["checkStatus"] == "FAIL"
    assert any(
        check["name"] == "CHAT_ANSWER_INTERNAL_TOKEN"
        and "placeholder" in check["reason"]
        for check in result["checks"]
    )


def test_check_k8s_runtime_env_rejects_localhost_service_urls() -> None:
    values = _valid_env_values()
    values["QDRANT_URL"] = "http://localhost:6333"
    values["EMBEDDING_BASE_URL"] = "http://127.0.0.1:8002"

    result = check_k8s_runtime_env.check_k8s_runtime_env(values)

    assert result["checkStatus"] == "FAIL"
    failed_names = {
        check["name"] for check in result["checks"] if check["status"] == "FAIL"
    }
    assert {"QDRANT_URL", "EMBEDDING_BASE_URL"} <= failed_names


def test_check_k8s_runtime_env_rejects_local_llm_runtime_values() -> None:
    values = _valid_env_values()
    values["LLM_PROVIDER"] = "openai_compatible"
    values["LLM_BASE_URL"] = "http://llm-service:8001/v1"
    values["LLM_MAX_TOKENS"] = "2048"
    values["LLM_REASONING_EFFORT"] = "medium"
    values["LLM_RESPONSE_CACHE_ENABLED"] = "false"
    values["ANSWER_MAX_CHARS"] = "2000"
    values["PROMPT_MAX_EVIDENCE_ITEMS"] = "5"
    values["PROMPT_MAX_DOCUMENT_SOURCES"] = "5"
    values["PROMPT_MAX_SUMMARY_CHARS"] = "700"
    values["PROMPT_MAX_TOTAL_CHARS"] = "6000"

    result = check_k8s_runtime_env.check_k8s_runtime_env(values)

    assert result["checkStatus"] == "FAIL"
    failed_names = {
        check["name"] for check in result["checks"] if check["status"] == "FAIL"
    }
    assert {
        "LLM_PROVIDER",
        "LLM_BASE_URL",
        "LLM_MAX_TOKENS",
        "LLM_REASONING_EFFORT",
        "LLM_RESPONSE_CACHE_ENABLED",
        "ANSWER_MAX_CHARS",
        "PROMPT_MAX_EVIDENCE_ITEMS",
        "PROMPT_MAX_DOCUMENT_SOURCES",
        "PROMPT_MAX_SUMMARY_CHARS",
        "PROMPT_MAX_TOTAL_CHARS",
    } <= failed_names


def test_check_k8s_runtime_env_rejects_missing_openai_runtime_values() -> None:
    values = _valid_env_values()
    values["LLM_API_KEY"] = ""
    values["OPENAI_API_KEY"] = ""
    values["LLM_MODEL"] = ""
    values["LLM_ALLOWED_MODELS"] = ""

    result = check_k8s_runtime_env.check_k8s_runtime_env(values)

    assert result["checkStatus"] == "FAIL"
    failed_names = {
        check["name"] for check in result["checks"] if check["status"] == "FAIL"
    }
    assert {"OPENAI_API_KEY", "LLM_MODEL", "LLM_ALLOWED_MODELS"} <= failed_names


def test_check_k8s_runtime_env_rejects_model_outside_allowlist() -> None:
    values = _valid_env_values()
    values["LLM_MODEL"] = "gpt-test"
    values["LLM_ALLOWED_MODELS"] = "gpt-other"

    result = check_k8s_runtime_env.check_k8s_runtime_env(values)

    assert result["checkStatus"] == "FAIL"
    assert any(
        check["name"] == "LLM_MODEL_ALLOWLIST"
        and check["status"] == "FAIL"
        and check["actual"] == "<set>"
        for check in result["checks"]
    )


def test_check_k8s_runtime_env_loads_example_file() -> None:
    result = check_k8s_runtime_env.check_k8s_runtime_env(
        check_k8s_runtime_env.load_env_values(
            Path("deploy/kubernetes/fastapi-chat-runtime.env.example")
        ),
        allow_placeholders=True,
    )

    assert result["checkStatus"] == "PASS"


def test_check_k8s_runtime_env_parses_stdin_style_env_output() -> None:
    raw_env_text = "\n".join(
        [
            "# comment",
            "IGNORED_LINE_WITHOUT_EQUALS",
            *[
                f"{key}={value}"
                for key, value in _valid_env_values().items()
            ],
        ]
    )

    result = check_k8s_runtime_env.check_k8s_runtime_env(
        check_k8s_runtime_env.parse_env_values(raw_env_text)
    )

    assert result["checkStatus"] == "PASS"


def test_check_k8s_runtime_env_main_accepts_stdin_env_values() -> None:
    stdout = StringIO()
    stdin = StringIO(
        "\n".join(
            f"{key}={value}"
            for key, value in _valid_env_values().items()
        )
    )

    exit_code = check_k8s_runtime_env.main(
        ["--env-file", "-", "--json"],
        stdout=stdout,
        stdin=stdin,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert '"checkStatus": "PASS"' in output
    assert "openai-secret-token" not in output
    assert '"actual": "<set>"' in output


def test_check_k8s_runtime_env_main_prints_json() -> None:
    stdout = StringIO()

    exit_code = check_k8s_runtime_env.main(
        ["--allow-placeholders", "--json"],
        stdout=stdout,
    )

    assert exit_code == 0
    assert '"checkStatus": "PASS"' in stdout.getvalue()
