from app.core.config import Settings
from app.features.chat.answer_output_policy import AnswerOutputPolicy
from app.features.chat.grounded_prompt_builder import GroundedPrompt, GroundedPromptBuilder
from app.features.chat.llm_client import LlmClient
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
    NO_GROUNDING_EVIDENCE,
)


class AnswerGenerationService:
    _insufficient_evidence_answer = (
        "현재 답변 생성을 위한 근거 조회 기능이 아직 연결되지 않았습니다. "
        "확인 가능한 근거 데이터가 부족해 답변할 수 없습니다."
    )

    def __init__(
        self,
        settings: Settings,
        prompt_builder: GroundedPromptBuilder | None = None,
        output_policy: AnswerOutputPolicy | None = None,
        llm_client: LlmClient | None = None,
    ) -> None:
        self.settings = settings
        self.prompt_builder = prompt_builder or GroundedPromptBuilder(settings)
        self.output_policy = output_policy or AnswerOutputPolicy()
        self.llm_client = llm_client or LlmClient(settings)

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
            return AnswerGenerationResult(
                answer="근거는 조회됐지만 LLM 답변 생성 기능이 아직 활성화되지 않았습니다.",
                was_generated=False,
                skipped_reason=LLM_DISABLED,
            )

        prompt = self.build_prompt(request, evidence_result, document_result)
        answer = await self.llm_client.generate(prompt)
        if not answer:
            return AnswerGenerationResult(
                answer="LLM 답변 생성 결과가 비어 있습니다.",
                was_generated=False,
                skipped_reason=LLM_EMPTY_ANSWER,
            )

        answer = self._ensure_source_titles(answer, evidence_result, document_result)
        output_security_result = self.output_policy.evaluate(
            answer,
            role=request.user.role,
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

        return AnswerGenerationResult(
            answer=answer,
            was_generated=True,
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
