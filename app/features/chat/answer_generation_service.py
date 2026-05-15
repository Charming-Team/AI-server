from app.core.config import Settings
from app.features.chat.grounded_prompt_builder import GroundedPrompt, GroundedPromptBuilder
from app.features.chat.llm_client import LlmClient
from app.features.chat.schemas import (
    AnswerGenerationResult,
    ChatAnswerRequest,
    DocumentSearchResult,
    EvidenceResult,
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
        llm_client: LlmClient | None = None,
    ) -> None:
        self.settings = settings
        self.prompt_builder = prompt_builder or GroundedPromptBuilder(settings)
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
                skipped_reason="No RDB Evidence or document sources are available.",
            )

        if not self.settings.llm_enabled:
            return AnswerGenerationResult(
                answer="근거는 조회됐지만 LLM 답변 생성 기능이 아직 활성화되지 않았습니다.",
                was_generated=False,
                skipped_reason="LLM is disabled.",
            )

        prompt = self.build_prompt(request, evidence_result, document_result)
        answer = await self.llm_client.generate(prompt)
        if not answer:
            return AnswerGenerationResult(
                answer="LLM 답변 생성 결과가 비어 있습니다.",
                was_generated=False,
                skipped_reason="LLM returned an empty answer.",
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
