import re

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
    LlmUsage,
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
    _answer_section_names = ("핵심 답변", "근거", "확인 필요")
    _internal_url_pattern = re.compile(
        r"(?:\s*,\s*|\s+)?(?<![:/])/[A-Za-z0-9][A-Za-z0-9/_-]*"
        r"(?:\?[A-Za-z0-9=&._%-]+)?"
    )
    _iso_datetime_pattern = re.compile(
        r"(?<!\d)"
        r"(?P<year>20\d{2})-(?P<month>\d{2})-(?P<day>\d{2})"
        r"[T ]"
        r"(?P<hour>\d{2}):(?P<minute>\d{2})"
        r"(?::\d{2}(?:\.\d+)?)?"
        r"(?:Z|[+-]\d{2}:?\d{2}| UTC)?"
        r"(?!\d)"
    )
    _iso_date_pattern = re.compile(
        r"(?<!\d)"
        r"(?P<year>20\d{2})-(?P<month>\d{2})-(?P<day>\d{2})"
        r"(?![\dT:])"
    )
    _boilerplate_sentence_terms = (
        "RDB",
        "QDRANT",
        "Qdrant",
        "문서 검색 근거",
        "근거에 기반",
        "변경 여부",
        "필요 시 조회",
        "조회해 주세요",
        "권한 범위 내에서",
        "말씀해 주세요",
        "알려주시면",
        "알려주시고",
    )
    _source_reference_prefixes = (
        "[RDB]",
        "[QDRANT]",
        "RDB 출처",
        "QDRANT 출처",
        "Qdrant 출처",
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

        try:
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
                    llm_usage=LlmUsage(promptTokens=0, completionTokens=0, totalTokens=0),
                )
            completion = await self.llm_client.generate_completion(prompt)
        except ChatExternalServiceError:
            answer = self.fallback_answer_builder.build(evidence_result, document_result)
            return self._build_output_checked_result(
                answer,
                role=request.user.role,
                was_generated=False,
                skipped_reason=LLM_UNAVAILABLE,
            )

        result = self._build_generated_result_from_llm_answer(
            completion.answer,
            role=request.user.role,
            evidence_result=evidence_result,
            document_result=document_result,
            llm_cache_hit=False,
            llm_usage=completion.usage,
        )
        if result.was_generated and result.security_result is None:
            self.llm_response_cache.put(cache_key, completion.answer)
        return result

    def _build_generated_result_from_llm_answer(
        self,
        answer: str,
        *,
        role: str,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
        llm_cache_hit: bool,
        llm_usage: LlmUsage | None,
    ) -> AnswerGenerationResult:
        if not answer:
            return AnswerGenerationResult(
                answer="LLM 답변 생성 결과가 비어 있습니다.",
                was_generated=False,
                skipped_reason=LLM_EMPTY_ANSWER,
            )

        answer = self._normalize_chat_answer(answer)
        return self._build_output_checked_result(
            answer,
            role=role,
            was_generated=True,
            llm_cache_hit=llm_cache_hit,
            llm_usage=llm_usage,
        )

    def _build_cache_key(self, prompt: GroundedPrompt) -> str:
        return build_llm_response_cache_key(
            prompt,
            provider=self.settings.llm_provider,
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            reasoning_effort=self.settings.llm_reasoning_effort,
        )

    def _build_output_checked_result(
        self,
        answer: str,
        *,
        role: str,
        was_generated: bool,
        llm_cache_hit: bool = False,
        llm_usage: LlmUsage | None = None,
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
                llm_usage=llm_usage,
                skipped_reason=ANSWER_BLOCKED_BY_OUTPUT_POLICY,
                security_result=output_security_result,
            )

        answer = self._limit_answer_length(answer)
        return AnswerGenerationResult(
            answer=answer,
            was_generated=was_generated,
            llm_cache_hit=llm_cache_hit,
            llm_usage=llm_usage,
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

    def _normalize_chat_answer(self, answer: str) -> str:
        normalized_lines: list[str] = []
        for line in answer.splitlines():
            stripped_line = line.strip()
            if not stripped_line:
                if normalized_lines and normalized_lines[-1] != "":
                    normalized_lines.append("")
                continue

            normalized_line = self._strip_answer_section_prefix(stripped_line)
            if not normalized_line:
                continue

            normalized_line = self._strip_leading_bullet(normalized_line)
            if self._is_source_reference_line(normalized_line):
                continue
            normalized_lines.append(normalized_line)

        normalized_answer = self._join_chat_answer_lines(normalized_lines)
        normalized_answer = self._normalize_datetime_text(normalized_answer)
        normalized_answer = self._remove_internal_urls(normalized_answer)
        normalized_answer = self._remove_boilerplate_sentences(normalized_answer)
        if normalized_answer:
            return normalized_answer
        return "확인된 내부 근거 기준으로 답변합니다."

    def _strip_answer_section_prefix(self, line: str) -> str:
        normalized_line = line.strip().strip("# ").strip()
        normalized_line = normalized_line.strip("*").strip()

        section_name = self._extract_answer_section_name(normalized_line)
        if section_name is not None:
            return ""

        for section in self._answer_section_names:
            for separator in (":", "："):
                prefix = f"{section}{separator}"
                if normalized_line.startswith(prefix):
                    return normalized_line.removeprefix(prefix).strip()
        return line.strip()

    def _strip_leading_bullet(self, line: str) -> str:
        stripped_line = line.strip()
        while stripped_line.startswith(("- ", "* ", "• ")):
            stripped_line = stripped_line[2:].strip()
        return stripped_line

    def _is_source_reference_line(self, line: str) -> bool:
        stripped_line = line.strip()
        return stripped_line.startswith(self._source_reference_prefixes)

    def _join_chat_answer_lines(self, lines: list[str]) -> str:
        return " ".join(line for line in lines if line).strip()

    def _normalize_datetime_text(self, answer: str) -> str:
        normalized_answer = self._iso_datetime_pattern.sub(
            lambda match: (
                f"{match.group('year')}.{match.group('month')}.{match.group('day')} "
                f"{match.group('hour')}:{match.group('minute')}"
            ),
            answer,
        )
        return self._iso_date_pattern.sub(
            lambda match: (
                f"{match.group('year')}.{match.group('month')}.{match.group('day')}"
            ),
            normalized_answer,
        )

    def _remove_internal_urls(self, answer: str) -> str:
        sanitized_answer = self._internal_url_pattern.sub("", answer)
        sanitized_answer = re.sub(r"\s+([,.])", r"\1", sanitized_answer)
        sanitized_answer = re.sub(r":\s*([,.])", r"\1", sanitized_answer)
        sanitized_answer = re.sub(r"\s{2,}", " ", sanitized_answer)
        sanitized_answer = sanitized_answer.replace(" ,", ",").replace(" .", ".")
        sanitized_answer = sanitized_answer.replace(": .", ".").replace(":,", ",")
        return sanitized_answer.strip()

    def _remove_boilerplate_sentences(self, answer: str) -> str:
        sentences = re.split(r"(?<=[.!?。])\s+", answer)
        filtered_sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
            and not any(term in sentence for term in self._boilerplate_sentence_terms)
        ]
        return " ".join(filtered_sentences).strip()

    def _extract_answer_section_name(self, line: str) -> str | None:
        candidate = line.strip().strip("# ").strip()
        candidate = candidate.strip("*").strip()
        candidate = candidate.rstrip(":：").strip()
        candidate = candidate.strip("*").strip()
        if candidate in self._answer_section_names:
            return candidate
        return None

    def _limit_answer_length(self, answer: str) -> str:
        max_chars = self.settings.answer_max_chars
        if len(answer) <= max_chars:
            return answer

        suffix = "\n\n... 답변 길이 제한으로 일부 내용이 생략되었습니다."
        if max_chars <= len(suffix):
            return answer[:max_chars]
        return f"{answer[: max_chars - len(suffix)]}{suffix}"
