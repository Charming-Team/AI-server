from app.core.config import Settings
from app.features.chat.skip_reasons import LLM_DISABLED
from app.schemas.health import ChatRuntimeMode


def build_chat_runtime_mode(settings: Settings) -> ChatRuntimeMode:
    grounding_sources = _build_enabled_grounding_sources(settings)
    return ChatRuntimeMode(
        api_prefix=settings.api_v1_prefix,
        grounding_mode=_build_grounding_mode(grounding_sources),
        answer_mode="LLM" if settings.llm_enabled else "FALLBACK",
        rag_search_mode=_build_rag_search_mode(settings),
        enabled_grounding_sources=grounding_sources,
        expected_llm_skipped_reason=None if settings.llm_enabled else LLM_DISABLED,
    )


def _build_enabled_grounding_sources(settings: Settings) -> list[str]:
    sources: list[str] = []
    if settings.rdb_evidence_enabled:
        sources.append("RDB_EVIDENCE")
    if settings.qdrant_search_enabled:
        sources.append("QDRANT")
    if settings.evidence_lookup_enabled:
        sources.append("SPRING_EVIDENCE")
    return sources


def _build_grounding_mode(grounding_sources: list[str]) -> str:
    if not grounding_sources:
        return "NONE"
    if grounding_sources == ["RDB_EVIDENCE"]:
        return "RDB_ONLY"
    if grounding_sources == ["QDRANT"]:
        return "QDRANT_ONLY"
    if grounding_sources == ["SPRING_EVIDENCE"]:
        return "SPRING_EVIDENCE_ONLY"
    if grounding_sources == ["RDB_EVIDENCE", "QDRANT"]:
        return "RDB_QDRANT"
    return "HYBRID"


def _build_rag_search_mode(settings: Settings) -> str:
    if not settings.qdrant_search_enabled:
        return "DISABLED"
    if not settings.embedding_enabled:
        return "MISSING_EMBEDDING"
    return "ENABLED"
