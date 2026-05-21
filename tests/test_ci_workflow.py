from pathlib import Path

WORKFLOW_PATH = Path(".github/workflows/docker-build-push.yml")


def test_docker_build_workflow_runs_python_quality_gates() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "python -m ruff check ." in workflow
    assert "python -m pytest" in workflow
    assert "python -m scripts.check_chat_runtime --preset full --json" in workflow


def test_docker_build_workflow_configures_chat_runtime_gate() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "CHAT_ANSWER_INTERNAL_TOKEN: ci-answer-token" in workflow
    assert "CHAT_RECOMMENDATION_INTERNAL_TOKEN: ci-recommendation-token" in workflow
    assert "DOCUMENT_INDEX_INTERNAL_TOKEN: ci-document-token" in workflow
    assert 'RDB_EVIDENCE_ENABLED: "true"' in workflow
    assert 'QDRANT_SEARCH_ENABLED: "true"' in workflow
    assert 'EMBEDDING_ENABLED: "true"' in workflow
    assert "QDRANT_COLLECTION: smap_internal_documents" in workflow


def test_docker_build_workflow_runs_on_project_branch_patterns() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert '- "feat/**"' in workflow
    assert '- "fix/**"' in workflow
    assert '- "chore/**"' in workflow
    assert '- "codex/**"' in workflow
