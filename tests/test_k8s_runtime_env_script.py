from io import StringIO
from pathlib import Path

from scripts import check_k8s_runtime_env


def _valid_env_values() -> dict[str, str]:
    return {
        "ENVIRONMENT": "kubernetes",
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
    }


def test_check_k8s_runtime_env_accepts_runtime_values() -> None:
    result = check_k8s_runtime_env.check_k8s_runtime_env(_valid_env_values())

    assert result["checkStatus"] == "PASS"
    assert result["failureCount"] == 0


def test_check_k8s_runtime_env_allows_placeholders_for_example_file() -> None:
    values = _valid_env_values()
    values["CHAT_ANSWER_INTERNAL_TOKEN"] = "__SET_BY_SECRET__"
    values["CHAT_RECOMMENDATION_INTERNAL_TOKEN"] = "__SET_BY_SECRET__"
    values["DOCUMENT_INDEX_INTERNAL_TOKEN"] = "__SET_BY_SECRET__"
    values["RDB_EVIDENCE_DSN"] = "__SET_BY_SECRET__"

    result = check_k8s_runtime_env.check_k8s_runtime_env(
        values,
        allow_placeholders=True,
    )

    assert result["checkStatus"] == "PASS"


def test_check_k8s_runtime_env_rejects_placeholders_for_actual_runtime() -> None:
    values = _valid_env_values()
    values["DOCUMENT_INDEX_INTERNAL_TOKEN"] = "__SET_BY_SECRET__"

    result = check_k8s_runtime_env.check_k8s_runtime_env(values)

    assert result["checkStatus"] == "FAIL"
    assert any(
        check["name"] == "DOCUMENT_INDEX_INTERNAL_TOKEN"
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


def test_check_k8s_runtime_env_loads_example_file() -> None:
    result = check_k8s_runtime_env.check_k8s_runtime_env(
        check_k8s_runtime_env.load_env_values(
            Path("deploy/kubernetes/fastapi-chat-runtime.env.example")
        ),
        allow_placeholders=True,
    )

    assert result["checkStatus"] == "PASS"


def test_check_k8s_runtime_env_main_prints_json() -> None:
    stdout = StringIO()

    exit_code = check_k8s_runtime_env.main(
        ["--allow-placeholders", "--json"],
        stdout=stdout,
    )

    assert exit_code == 0
    assert '"checkStatus": "PASS"' in stdout.getvalue()
