import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_accepts_chat_cost_guardrail_limits() -> None:
    settings = Settings(
        qdrant_top_k=20,
        qdrant_score_threshold=1.0,
        llm_max_tokens=4096,
        prompt_max_evidence_items=20,
        prompt_max_document_sources=20,
    )

    assert settings.qdrant_top_k == 20
    assert settings.qdrant_score_threshold == 1.0
    assert settings.llm_max_tokens == 4096
    assert settings.prompt_max_evidence_items == 20
    assert settings.prompt_max_document_sources == 20


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("qdrant_top_k", 0),
        ("qdrant_top_k", 21),
        ("qdrant_score_threshold", -0.1),
        ("qdrant_score_threshold", 1.1),
        ("llm_max_tokens", 0),
        ("llm_max_tokens", 4097),
        ("prompt_max_evidence_items", -1),
        ("prompt_max_evidence_items", 21),
        ("prompt_max_document_sources", -1),
        ("prompt_max_document_sources", 21),
    ],
)
def test_settings_rejects_chat_cost_guardrail_violations(
    field_name: str,
    value: int | float,
) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field_name: value})
