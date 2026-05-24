from pathlib import Path

import yaml

WORKFLOW_PATH = Path(".github/workflows/docker-build-push.yml")


def _load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _build_steps() -> list[dict]:
    workflow = _load_workflow()
    return workflow["jobs"]["build-and-push"]["steps"]


def _step_by_name(name: str) -> dict:
    for step in _build_steps():
        if step.get("name") == name:
            return step
    raise AssertionError(f"Workflow step not found: {name}")


def test_docker_build_workflow_runs_python_quality_gates() -> None:
    steps = _build_steps()
    runs = [step.get("run") for step in steps]

    assert "python -m ruff check ." in runs
    assert "python -m pytest" in runs
    assert (
        "python -m scripts.check_chat_runtime --preset full "
        "--answer-api-max-llm-total-tokens 2000 --json"
    ) in runs


def test_docker_build_workflow_configures_chat_runtime_gate() -> None:
    release_gate_step = _step_by_name("Run chatbot release gate")
    env = release_gate_step["env"]

    assert env["CHAT_ANSWER_INTERNAL_TOKEN"] == "ci-answer-token"
    assert env["CHAT_RECOMMENDATION_INTERNAL_TOKEN"] == "ci-recommendation-token"
    assert "DOCUMENT_INDEX_INTERNAL_TOKEN" not in env
    assert env["RDB_EVIDENCE_ENABLED"] == "true"
    assert env["QDRANT_SEARCH_ENABLED"] == "true"
    assert env["EMBEDDING_ENABLED"] == "true"
    assert env["EMBEDDING_DIMENSION"] == "1024"
    assert env["QDRANT_COLLECTION"] == "smap_internal_documents"
    assert env["LLM_ENABLED"] == "true"
    assert env["LLM_PROVIDER"] == "openai_compatible"
    assert env["LLM_BASE_URL"] == "http://localhost:8001/v1"
    assert env["LLM_MODEL"] == "local-open-source-model"
    assert env["LLM_MAX_TOKENS"] == "512"
    assert env["LLM_RESPONSE_CACHE_ENABLED"] == "true"
    assert env["LLM_RESPONSE_CACHE_TTL_SECONDS"] == "60.0"
    assert env["LLM_RESPONSE_CACHE_MAX_ENTRIES"] == "128"
    assert env["ANSWER_MAX_CHARS"] == "1600"
    assert env["PROMPT_MAX_EVIDENCE_ITEMS"] == "3"
    assert env["PROMPT_MAX_DOCUMENT_SOURCES"] == "3"
    assert env["PROMPT_MAX_SUMMARY_CHARS"] == "360"
    assert env["PROMPT_MAX_DATA_CHARS"] == "500"
    assert env["PROMPT_MAX_TOTAL_CHARS"] == "4000"


def test_docker_build_workflow_runs_on_project_branch_patterns() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert '- "dev/**"' in workflow
    assert '- "feat/**"' in workflow
    assert '- "fix/**"' in workflow
    assert '- "chore/**"' in workflow
