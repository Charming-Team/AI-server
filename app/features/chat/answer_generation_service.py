from app.core.config import Settings
from app.features.chat.answer_output_policy import AnswerOutputPolicy
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.grounded_fallback_answer_builder import (
    GroundedFallbackAnswerBuilder,
)
from app.features.chat.grounded_prompt_builder import GroundedPrompt, GroundedPromptBuilder
from app.features.chat.llm_client import LlmClient, validate_llm_settings
from app.features.chat.llm_response_cache import (
    LlmResponseCache,
    build_llm_response_cache_key,
)
from app.features.chat.schemas import (
    AnswerGenerationResult,
    ChatAnswerRequest,
    DocumentSearchResult,
    EvidenceResult,
)
from app.features.chat.skip_reasons import (
    ANSWER_BLOCKED_BY_OUTPUT_POLICY,
    LLM_DISABLED,
    LLM_EMPTY_ANSWER,
    LLM_UNAVAILABLE,
    NO_GROUNDING_EVIDENCE,
)


class AnswerGenerationService:
    _insufficient_evidence_answer = (
        "질문과 관련해 확인 가능한 내부 근거를 찾지 못했습니다. "
        "추측성 답변은 제공하지 않습니다.\n\n"
        "자재, 라인, 주문 번호, 기간 같은 기준을 포함해 다시 질문하거나 "
        "관련 RDB 데이터와 Qdrant 문서 등록 상태를 확인해 주세요."
    )

    def __init__(
        self,
        settings: Settings,
        prompt_builder: GroundedPromptBuilder | None = None,
        fallback_answer_builder: GroundedFallbackAnswerBuilder | None = None,
        output_policy: AnswerOutputPolicy | None = None,
        llm_client: LlmClient | None = None,
        llm_response_cache: LlmResponseCache | None = None,
    ) -> None:
        self.settings = settings
        self.prompt_builder = prompt_builder or GroundedPromptBuilder(settings)
        self.fallback_answer_builder = (
            fallback_answer_builder or GroundedFallbackAnswerBuilder()
        )
        self.output_policy = output_policy or AnswerOutputPolicy()
        self.llm_client = llm_client or LlmClient(settings)
        self.llm_response_cache = llm_response_cache or LlmResponseCache(
            enabled=settings.llm_response_cache_enabled,
            ttl_seconds=settings.llm_response_cache_ttl_seconds,
            max_entries=settings.llm_response_cache_max_entries,
        )

    async def generate_answer(
        self,
        request: ChatAnswerRequest,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> AnswerGenerationResult:
        if not self._has_grounding(evidence_result, document_result):
            return AnswerGenerationResult(
                answer=self._insufficient_evidence_answer,
                was_generated=False,
                skipped_reason=NO_GROUNDING_EVIDENCE,
            )

        if not self.settings.llm_enabled:
            answer = self.fallback_answer_builder.build(evidence_result, document_result)
            return self._build_output_checked_result(
                answer,
                role=request.user.role,
                was_generated=False,
                skipped_reason=LLM_DISABLED,
            )

        validate_llm_settings(self.settings)
        prompt = self.build_prompt(request, evidence_result, document_result)
        cache_key = self._build_cache_key(prompt)
        if cached_answer := self.llm_response_cache.get(cache_key):
            return self._build_generated_result_from_llm_answer(
                cached_answer,
                role=request.user.role,
                evidence_result=evidence_result,
                document_result=document_result,
                llm_cache_hit=True,
            )

        try:
            answer = await self.llm_client.generate(prompt)
        except ChatExternalServiceError:
            answer = self.fallback_answer_builder.build(evidence_result, document_result)
            return self._build_output_checked_result(
                answer,
                role=request.user.role,
                was_generated=False,
                skipped_reason=LLM_UNAVAILABLE,
            )

        result = self._build_generated_result_from_llm_answer(
            answer,
            role=request.user.role,
            evidence_result=evidence_result,
            document_result=document_result,
            llm_cache_hit=False,
        )
        if result.was_generated and result.security_result is None:
            self.llm_response_cache.put(cache_key, answer)
        return result

    def _build_generated_result_from_llm_answer(
        self,
        answer: str,
        *,
        role: str,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
        llm_cache_hit: bool,
    ) -> AnswerGenerationResult:
        if not answer:
            return AnswerGenerationResult(
                answer="LLM 답변 생성 결과가 비어 있습니다.",
                was_generated=False,
                skipped_reason=LLM_EMPTY_ANSWER,
            )

        answer = self._ensure_answer_sections(
            answer,
            evidence_result,
            document_result,
        )
        answer = self._ensure_source_titles(answer, evidence_result, document_result)
        return self._build_output_checked_result(
            answer,
            role=role,
            was_generated=True,
            llm_cache_hit=llm_cache_hit,
        )

    def _build_cache_key(self, prompt: GroundedPrompt) -> str:
        return build_llm_response_cache_key(
            prompt,
            provider=self.settings.llm_provider,
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
        )

    def _build_output_checked_result(
        self,
        answer: str,
        *,
        role: str,
        was_generated: bool,
        llm_cache_hit: bool = False,
        skipped_reason: str | None = None,
    ) -> AnswerGenerationResult:
        output_security_result = self.output_policy.evaluate(
            answer,
            role=role,
        )
        if output_security_result is not None:
            return AnswerGenerationResult(
                answer=(
                    "보안상 생성된 답변을 제공할 수 없습니다. "
                    "업무 데이터에 대한 질문으로 다시 요청해 주세요."
                ),
                was_generated=False,
                skipped_reason=ANSWER_BLOCKED_BY_OUTPUT_POLICY,
                security_result=output_security_result,
            )

        answer = self._limit_answer_length(answer)
        return AnswerGenerationResult(
            answer=answer,
            was_generated=was_generated,
            llm_cache_hit=llm_cache_hit,
            skipped_reason=skipped_reason,
        )

    def build_prompt(
        self,
        request: ChatAnswerRequest,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> GroundedPrompt:
        return self.prompt_builder.build(request, evidence_result, document_result)

    def _has_grounding(
        self,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> bool:
        return evidence_result.has_evidence or bool(document_result.sources)

    def _ensure_source_titles(
        self,
        answer: str,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> str:
        titles = self._collect_source_titles(evidence_result, document_result)
        if not titles:
            return answer

        normalized_answer = answer.casefold()
        if any(title.casefold() in normalized_answer for title in titles):
            return answer

        return f"{answer}\n\n참조 근거: {', '.join(titles[:3])}"

    def _collect_source_titles(
        self,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> list[str]:
        titles: list[str] = []
        seen_titles: set[str] = set()
        for title in [
            *(item.title for item in evidence_result.items),
            *(source.title for source in document_result.sources),
        ]:
            normalized_title = title.casefold()
            if normalized_title in seen_titles:
                continue
            seen_titles.add(normalized_title)
            titles.append(title)
        return titles

    def _ensure_answer_sections(
        self,
        answer: str,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> str:
        if self._has_required_sections(answer):
            return answer

        source_titles = self._collect_source_titles(evidence_result, document_result)
        if source_titles:
            evidence_lines = "\n".join(f"- {title}" for title in source_titles[:3])
        else:
            evidence_lines = "- 제공된 내부 근거"

        return "\n\n".join(
            [
                f"핵심 답변:\n{answer.strip()}",
                f"근거:\n{evidence_lines}",
                (
                    "확인 필요:\n"
                    "- 위 근거에 포함되지 않은 수치, 원인, 조치는 추가 확인이 필요합니다."
                ),
            ]
        )

    def _has_required_sections(self, answer: str) -> bool:
        normalized_answer = answer.casefold()
        return all(
            section in normalized_answer
            for section in ("핵심 답변", "근거", "확인 필요")
        )

    def _limit_answer_length(self, answer: str) -> str:
        max_chars = self.settings.answer_max_chars
        if len(answer) <= max_chars:
            return answer

        suffix = "\n\n... 답변 길이 제한으로 일부 내용이 생략되었습니다."
        if max_chars <= len(suffix):
            return answer[:max_chars]
        return f"{answer[: max_chars - len(suffix)]}{suffix}"
