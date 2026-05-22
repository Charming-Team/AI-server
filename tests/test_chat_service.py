from datetime import datetime

import anyio

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.schemas import (
    AnswerGenerationResult,
    ChatAnswerRequest,
    ChatAnswerResponse,
    ChatErrorCode,
    ChatIntent,
    ChatSource,
    ChatUserContext,
    DocumentSearchResult,
    EvidenceItem,
    EvidenceResult,
    SecurityResult,
    SecurityStatus,
)
from app.features.chat.service import ChatService

BLOCKED_GENERATED_ANSWER = (
    "보안상 생성된 답변을 제공할 수 없습니다. "
    "업무 데이터에 대한 질문으로 다시 요청해 주세요."
)
SENSITIVE_OUTPUT_REASON = (
    "생성 답변에 민감 정보 또는 내부 설정 정보가 포함된 것으로 판단되었습니다."
)


class FakeEvidenceService:
    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        return EvidenceResult(
            intent=intent,
            basisTime=request.requested_at,
            items=[
                EvidenceItem(
                    type="REPORT",
                    title="월간 생산 리스크",
                    summary="자재 부족이 주요 리스크입니다.",
                    source="reports",
                )
            ],
        )


class FakeDocumentSearchService:
    def __init__(
        self,
        sources: list[ChatSource] | None = None,
        was_searched: bool = False,
        skipped_reason: str | None = None,
    ) -> None:
        self.sources = sources or []
        self.was_searched = was_searched
        self.skipped_reason = skipped_reason

    async def search(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> DocumentSearchResult:
        return DocumentSearchResult(
            was_searched=self.was_searched,
            sources=self.sources,
            skipped_reason=self.skipped_reason,
        )


class FakeFailingDocumentSearchService:
    def __init__(self, code: ChatErrorCode, message: str) -> None:
        self.code = code
        self.message = message

    async def search(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> DocumentSearchResult:
        raise ChatExternalServiceError(
            status_code=503,
            code=self.code,
            message=self.message,
        )


class FakeGeneratedAnswerGenerationService:
    async def generate_answer(
        self,
        request: ChatAnswerRequest,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> AnswerGenerationResult:
        return AnswerGenerationResult(
            answer="근거에 따르면 자재 부족이 주요 리스크입니다.",
            was_generated=True,
        )


class FakeCapturingAnswerGenerationService:
    def __init__(self) -> None:
        self.evidence_result: EvidenceResult | None = None
        self.document_result: DocumentSearchResult | None = None

    async def generate_answer(
        self,
        request: ChatAnswerRequest,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> AnswerGenerationResult:
        self.evidence_result = evidence_result
        self.document_result = document_result
        return AnswerGenerationResult(
            answer="근거에 따르면 납기 위험이 있습니다.",
            was_generated=True,
        )


class FakeBlockedAnswerGenerationService:
    async def generate_answer(
        self,
        request: ChatAnswerRequest,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> AnswerGenerationResult:
        return AnswerGenerationResult(
            answer=BLOCKED_GENERATED_ANSWER,
            was_generated=False,
            skipped_reason="생성 답변이 출력 보안 정책에 의해 차단되었습니다.",
            security_result=SecurityResult(
                status=SecurityStatus.BLOCKED_SENSITIVE_REQUEST,
                reason="생성 답변에 민감 정보 또는 내부 설정 정보가 포함된 것으로 판단되었습니다.",
            ),
        )


class FakeAuditLogger:
    def __init__(self) -> None:
        self.requests: list[ChatAnswerRequest] = []
        self.responses: list[ChatAnswerResponse] = []

    def log_answer_response(
        self,
        request: ChatAnswerRequest,
        response: ChatAnswerResponse,
    ) -> None:
        self.requests.append(request)
        self.responses.append(response)


class FakeUnexpectedEvidenceService:
    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        raise AssertionError("비활성 계정은 RDB Evidence 조회를 호출하면 안 됩니다.")


class FakeFinancialEvidenceService:
    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        return EvidenceResult(
            intent=intent,
            basisTime=request.requested_at,
            items=[
                EvidenceItem(
                    type="ORDER",
                    title="ORD-202605-001 납기 위험",
                    summary="납기 지연 위험 등급은 WARNING입니다.",
                    source="ai_prediction_results",
                    data={
                        "riskLevel": "WARNING",
                        "contractAmount": 12000000,
                        "latePenaltyAmount": 500000,
                        "recommendedAction": "생산 순서 조정",
                        "nested": {
                            "costChangeAmount": 300000,
                            "lineCode": "LINE-A01",
                        },
                    },
                )
            ],
        )


class FakeScopedEvidenceService:
    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        return EvidenceResult(
            intent=intent,
            basisTime=request.requested_at,
            items=[
                EvidenceItem(
                    type="ORDER",
                    title="허용된 납기 위험",
                    summary="현재 사용자 범위에 포함됩니다.",
                    source="ai_prediction_results",
                    allowedRoles=["EXECUTIVE"],
                    companyName="S-MAP",
                ),
                EvidenceItem(
                    type="ORDER",
                    title="다른 Role 납기 위험",
                    summary="현재 사용자 범위 밖입니다.",
                    source="ai_prediction_results",
                    allowedRoles=["MANUFACTURING_MANAGER"],
                    companyName="S-MAP",
                ),
                EvidenceItem(
                    type="ORDER",
                    title="다른 회사명 납기 위험",
                    summary="회사명은 로그용이므로 접근 제어 조건으로 쓰지 않습니다.",
                    source="ai_prediction_results",
                    allowedRoles=["EXECUTIVE"],
                    companyName="OTHER",
                ),
            ],
        )


def _build_request(
    status: str = "ACTIVE",
    role: str = "EXECUTIVE",
    question: str = "최근 보고서 요약해줘",
) -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=10,
        messageId=24,
        user=ChatUserContext(
            userId=1,
            role=role,
            companyName="S-MAP",
            status=status,
        ),
        question=question,
        requestedAt=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
    )


def test_chat_service_blocks_inactive_user_status_before_evidence_lookup() -> None:
    service = ChatService(Settings())
    service.evidence_service = FakeUnexpectedEvidenceService()
    audit_logger = FakeAuditLogger()
    service.audit_logger = audit_logger

    response = anyio.run(service.create_answer, _build_request(status="SUSPENDED"))

    assert response.security_result.status == SecurityStatus.BLOCKED_UNAUTHORIZED
    assert response.security_result.code == "CHAT_SECURITY_004"
    assert response.security_result.reason == "ACTIVE 상태 사용자만 챗봇을 사용할 수 있습니다."
    assert response.answer == "현재 계정 상태로는 챗봇을 사용할 수 없습니다."
    assert response.model_result.used_rdb_evidence is False
    assert response.model_result.used_vector_search is False
    assert response.model_result.used_llm_generation is False
    assert audit_logger.requests == [_build_request(status="SUSPENDED")]
    assert audit_logger.responses == [response]


def test_chat_service_uses_answer_output_security_result() -> None:
    service = ChatService(Settings())
    service.evidence_service = FakeEvidenceService()
    service.document_search_service = FakeDocumentSearchService()
    service.answer_generation_service = FakeBlockedAnswerGenerationService()

    response = anyio.run(service.create_answer, _build_request())

    assert response.security_result.status == SecurityStatus.BLOCKED_SENSITIVE_REQUEST
    assert response.security_result.reason == SENSITIVE_OUTPUT_REASON
    assert response.answer == BLOCKED_GENERATED_ANSWER
    assert (
        response.model_result.llm_generation_skipped_reason
        == "생성 답변이 출력 보안 정책에 의해 차단되었습니다."
    )


def test_chat_service_builds_detailed_model_result_counts() -> None:
    service = ChatService(Settings())
    audit_logger = FakeAuditLogger()
    service.evidence_service = FakeEvidenceService()
    service.document_search_service = FakeDocumentSearchService(
        was_searched=True,
        sources=[
            ChatSource(
                sourceType="REPORT",
                title="월간 생산 리스크 보고서",
                summary="자재 부족이 주요 리스크입니다.",
            ),
            ChatSource(
                sourceType="COMPANY_INFO",
                title="회사 운영 기준",
                summary="생산계획 우선순위 기준입니다.",
            ),
        ],
    )
    service.answer_generation_service = FakeGeneratedAnswerGenerationService()
    service.audit_logger = audit_logger

    request = _build_request()
    response = anyio.run(service.create_answer, request)

    assert response.model_result.used_vector_search is True
    assert response.model_result.used_rdb_evidence is True
    assert response.model_result.used_llm_generation is True
    assert response.model_result.rdb_evidence_count == 1
    assert response.model_result.document_source_count == 2
    assert response.model_result.evidence_count == 3
    assert response.model_result.vector_search_skipped_reason is None
    assert response.model_result.llm_generation_skipped_reason is None
    assert audit_logger.requests == [request]
    assert audit_logger.responses == [response]


def test_chat_service_uses_latest_grounding_basis_time() -> None:
    service = ChatService(Settings())
    service.evidence_service = FakeEvidenceService()
    service.document_search_service = FakeDocumentSearchService(
        was_searched=True,
        sources=[
            ChatSource(
                sourceType="REPORT",
                title="월간 생산 리스크 보고서",
                summary="자재 부족과 LINE-A01 병목이 주요 리스크입니다.",
                basisTime=datetime.fromisoformat("2026-05-12T11:00:00+09:00"),
                sourceOrigin="QDRANT",
            )
        ],
    )
    service.answer_generation_service = FakeGeneratedAnswerGenerationService()

    response = anyio.run(service.create_answer, _build_request())

    assert response.basis_time == datetime.fromisoformat("2026-05-12T11:00:00+09:00")


def test_chat_service_sanitizes_operator_financial_evidence_before_llm() -> None:
    service = ChatService(Settings())
    answer_generation_service = FakeCapturingAnswerGenerationService()
    service.evidence_service = FakeFinancialEvidenceService()
    service.document_search_service = FakeDocumentSearchService()
    service.answer_generation_service = answer_generation_service

    response = anyio.run(
        service.create_answer,
        _build_request(
            role="OPERATOR",
            question="현재 납기 위험 높은 주문 알려줘",
        ),
    )

    assert response.answer == "근거에 따르면 납기 위험이 있습니다."
    assert answer_generation_service.evidence_result is not None
    evidence_data = answer_generation_service.evidence_result.items[0].data
    assert evidence_data["riskLevel"] == "WARNING"
    assert evidence_data["recommendedAction"] == "생산 순서 조정"
    assert evidence_data["nested"] == {"lineCode": "LINE-A01"}
    assert "contractAmount" not in evidence_data
    assert "latePenaltyAmount" not in evidence_data


def test_chat_service_sanitizes_operator_financial_document_sources_before_llm() -> None:
    service = ChatService(Settings())
    answer_generation_service = FakeCapturingAnswerGenerationService()
    service.evidence_service = FakeEvidenceService()
    service.document_search_service = FakeDocumentSearchService(
        was_searched=True,
        sources=[
            ChatSource(
                sourceType="REPORT",
                title="납기 지연 패널티 보고서",
                summary="계약 금액과 패널티 금액을 함께 검토합니다.",
                url="/reports/finance?mode=read",
                sourceOrigin="QDRANT",
            ),
            ChatSource(
                sourceType="REPORT",
                title="월간 생산 리스크 보고서",
                summary="자재 부족과 LINE-A01 병목이 주요 리스크입니다.",
                url="/reports/20?mode=read",
                sourceOrigin="QDRANT",
            ),
        ],
    )
    service.answer_generation_service = answer_generation_service

    response = anyio.run(
        service.create_answer,
        _build_request(
            role="OPERATOR",
            question="최근 생산 리스크 보고서를 조회해줘",
        ),
    )

    assert answer_generation_service.document_result is not None
    assert [
        source.title for source in answer_generation_service.document_result.sources
    ] == ["월간 생산 리스크 보고서"]
    assert response.model_result.document_source_count == 1
    assert [source.title for source in response.sources] == [
        "월간 생산 리스크",
        "월간 생산 리스크 보고서",
    ]
    assert [url.url for url in response.urls] == ["/reports/20?mode=read"]


def test_chat_service_filters_allowed_roles_before_llm_and_ignores_company_name() -> None:
    service = ChatService(Settings())
    answer_generation_service = FakeCapturingAnswerGenerationService()
    service.evidence_service = FakeScopedEvidenceService()
    service.document_search_service = FakeDocumentSearchService()
    service.answer_generation_service = answer_generation_service

    response = anyio.run(
        service.create_answer,
        _build_request(
            question="현재 납기 위험 높은 주문 알려줘",
        ),
    )

    assert response.answer == "근거에 따르면 납기 위험이 있습니다."
    assert answer_generation_service.evidence_result is not None
    assert [item.title for item in answer_generation_service.evidence_result.items] == [
        "허용된 납기 위험",
        "다른 회사명 납기 위험",
    ]
    assert response.model_result.rdb_evidence_count == 2


def test_chat_service_builds_model_result_skipped_reasons() -> None:
    service = ChatService(Settings())
    service.evidence_service = FakeEvidenceService()
    service.document_search_service = FakeDocumentSearchService(
        skipped_reason="임베딩 기능이 비활성화되어 있습니다."
    )
    service.answer_generation_service = FakeBlockedAnswerGenerationService()

    response = anyio.run(service.create_answer, _build_request())

    assert response.model_result.used_vector_search is False
    assert response.model_result.used_llm_generation is False
    assert (
        response.model_result.vector_search_skipped_reason
        == "임베딩 기능이 비활성화되어 있습니다."
    )
    assert (
        response.model_result.llm_generation_skipped_reason
        == "생성 답변이 출력 보안 정책에 의해 차단되었습니다."
    )


def test_chat_service_degrades_to_rdb_evidence_when_qdrant_search_fails() -> None:
    service = ChatService(Settings())
    answer_generation_service = FakeCapturingAnswerGenerationService()
    service.evidence_service = FakeEvidenceService()
    service.document_search_service = FakeFailingDocumentSearchService(
        code=ChatErrorCode.CHAT_QDRANT_004,
        message="Qdrant 검색에 실패했습니다.",
    )
    service.answer_generation_service = answer_generation_service

    response = anyio.run(service.create_answer, _build_request())

    assert response.answer == "근거에 따르면 납기 위험이 있습니다."
    assert response.security_result.status == SecurityStatus.PASSED
    assert response.model_result.used_rdb_evidence is True
    assert response.model_result.used_vector_search is True
    assert response.model_result.document_source_count == 0
    assert response.model_result.evidence_count == 1
    assert response.model_result.vector_search_skipped_reason == "Qdrant 검색에 실패했습니다."
    assert answer_generation_service.document_result == DocumentSearchResult(
        was_searched=True,
        skipped_reason="Qdrant 검색에 실패했습니다.",
    )


def test_chat_service_marks_embedding_failure_without_vector_search() -> None:
    service = ChatService(Settings())
    service.evidence_service = FakeEvidenceService()
    service.document_search_service = FakeFailingDocumentSearchService(
        code=ChatErrorCode.CHAT_EMBEDDING_004,
        message="임베딩 서버 호출에 실패했습니다.",
    )
    service.answer_generation_service = FakeGeneratedAnswerGenerationService()

    response = anyio.run(service.create_answer, _build_request())

    assert response.security_result.status == SecurityStatus.PASSED
    assert response.model_result.used_rdb_evidence is True
    assert response.model_result.used_vector_search is False
    assert (
        response.model_result.vector_search_skipped_reason
        == "임베딩 서버 호출에 실패했습니다."
    )
