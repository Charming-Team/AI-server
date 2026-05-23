from datetime import datetime

from app.core.config import Settings
from app.features.chat.grounded_prompt_builder import GroundedPromptBuilder
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatIntent,
    ChatSource,
    ChatUserContext,
    DocumentSearchResult,
    EvidenceItem,
    EvidenceResult,
)


def _build_request(role: str = "EXECUTIVE") -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=10,
        messageId=24,
        user=ChatUserContext(
            userId=1,
            role=role,
            companyName="S-MAP",
            status="ACTIVE",
        ),
        question="최근 납기 위험과 자재 부족 근거를 요약해줘",
        requestedAt=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
    )


def test_grounded_prompt_builder_includes_internal_grounding_rules() -> None:
    builder = GroundedPromptBuilder()
    request = _build_request()
    evidence_result = EvidenceResult(
        intent=ChatIntent.DELIVERY_RISK,
        basisTime=request.requested_at,
        items=[],
    )
    document_result = DocumentSearchResult(sources=[])

    prompt = builder.build(request, evidence_result, document_result)

    assert "제공된 내부 근거만 사용" in prompt.system_prompt
    assert "웹 검색" in prompt.system_prompt
    assert "일반 상식" in prompt.system_prompt
    assert "핵심 답변, 근거, 확인 필요 순서" in prompt.system_prompt
    assert "사용자 역할:\nEXECUTIVE" in prompt.user_prompt
    assert "역할별 응답 제한:" in prompt.user_prompt
    assert "RDB 근거:\n없음" in prompt.user_prompt
    assert "문서 검색 근거:\n없음" in prompt.user_prompt
    assert "수치, 상태, 날짜는 RDB 근거 또는 문서 검색 근거" in prompt.user_prompt
    assert "답변 형식은 핵심 답변, 근거, 확인 필요 순서" in prompt.user_prompt


def test_grounded_prompt_builder_includes_operator_role_constraints() -> None:
    builder = GroundedPromptBuilder()
    request = _build_request(role="OPERATOR")
    evidence_result = EvidenceResult(
        intent=ChatIntent.REPORT_LOOKUP,
        basisTime=request.requested_at,
        items=[],
    )
    document_result = DocumentSearchResult(sources=[])

    prompt = builder.build(request, evidence_result, document_result)

    assert "사용자 역할:\nOPERATOR" in prompt.user_prompt
    assert "OPERATOR에게는 금액, 계약, 패널티, 비용, 매출, 수익" in (
        prompt.user_prompt
    )
    assert "비금액성 보고서 근거만 요약" in prompt.user_prompt


def test_grounded_prompt_builder_formats_evidence_and_document_sources() -> None:
    builder = GroundedPromptBuilder()
    request = _build_request()
    evidence_result = EvidenceResult(
        intent=ChatIntent.DELIVERY_RISK,
        basisTime=request.requested_at,
        items=[
            EvidenceItem(
                type="ORDER",
                title="ORD-202605-001 납기 지연 위험",
                summary="납기일이 임박했고 현재 계획 상태는 DELAYED입니다.",
                url="/orders/1001",
                source="customer_orders",
                referenceId=1001,
                data={"orderNo": "ORD-202605-001", "riskLevel": "WARNING"},
            )
        ],
    )
    document_result = DocumentSearchResult(
        sources=[
            ChatSource(
                sourceType="REPORT",
                title="2026년 5월 생산 리스크 보고서",
                summary="자재 부족과 LINE-A01 병목이 주요 리스크입니다.",
                url="/reports/20",
                referenceId=20,
                source="report-202605:summary",
                basisTime=datetime.fromisoformat("2026-05-12T11:00:00+09:00"),
                sourceOrigin="QDRANT",
                relevanceScore=0.91823,
            )
        ]
    )

    prompt = builder.build(request, evidence_result, document_result)

    assert "ORD-202605-001 납기 지연 위험" in prompt.user_prompt
    assert '"orderNo": "ORD-202605-001"' in prompt.user_prompt
    assert "2026년 5월 생산 리스크 보고서" in prompt.user_prompt
    assert "report-202605:summary" in prompt.user_prompt
    assert "근거 원천: QDRANT" in prompt.user_prompt
    assert "관련도 점수: 0.9182" in prompt.user_prompt
    assert "기준 시각: 2026-05-12T11:00:00+09:00" in prompt.user_prompt


def test_grounded_prompt_builder_sanitizes_source_urls_before_prompt() -> None:
    builder = GroundedPromptBuilder()
    request = _build_request()
    evidence_result = EvidenceResult(
        intent=ChatIntent.REPORT_LOOKUP,
        basisTime=request.requested_at,
        items=[
            EvidenceItem(
                type="REPORT",
                title="외부 URL 보고서 근거",
                summary="URL은 프롬프트에 그대로 들어가면 안 됩니다.",
                url="https://evil.example/reports/20",
                source="reports",
            ),
            EvidenceItem(
                type="REPORT",
                title="내부 URL 보고서 근거",
                summary="내부 상대 경로만 프롬프트에 남습니다.",
                url=" /reports/20?mode=read ",
                source="reports",
            ),
        ],
    )
    document_result = DocumentSearchResult(
        sources=[
            ChatSource(
                sourceType="REPORT",
                title="스크립트 URL 문서 근거",
                summary="허용되지 않은 URL은 제외됩니다.",
                url="javascript:alert(1)",
            ),
            ChatSource(
                sourceType="REPORT",
                title="내부 URL 문서 근거",
                summary="허용된 URL만 포함됩니다.",
                url="/reports/21?mode=read",
            ),
        ]
    )

    prompt = builder.build(request, evidence_result, document_result)

    assert "https://evil.example" not in prompt.user_prompt
    assert "javascript:alert" not in prompt.user_prompt
    assert "URL: /reports/20?mode=read" in prompt.user_prompt
    assert "URL: /reports/21?mode=read" in prompt.user_prompt


def test_grounded_prompt_builder_sanitizes_data_urls_before_prompt() -> None:
    builder = GroundedPromptBuilder()
    request = _build_request()
    evidence_result = EvidenceResult(
        intent=ChatIntent.REPORT_LOOKUP,
        basisTime=request.requested_at,
        items=[
            EvidenceItem(
                type="REPORT",
                title="월간 생산 리스크 보고서",
                summary="원천 데이터 URL도 내부 경로만 프롬프트에 남습니다.",
                source="reports",
                data={
                    "reportUrl": "/reports/20?mode=read",
                    "externalUrl": "https://evil.example/reports/20",
                    "nested": {
                        "detailUrl": " /lines/1?mode=read ",
                        "scriptUrl": "javascript:alert(1)",
                    },
                    "relatedUrls": [
                        "/materials/11?mode=read",
                        "https://evil.example/materials/11",
                    ],
                },
            )
        ],
    )

    prompt = builder.build(
        request,
        evidence_result,
        DocumentSearchResult(sources=[]),
    )

    assert "https://evil.example" not in prompt.user_prompt
    assert "javascript:alert" not in prompt.user_prompt
    assert '"reportUrl": "/reports/20?mode=read"' in prompt.user_prompt
    assert '"detailUrl": "/lines/1?mode=read"' in prompt.user_prompt
    assert '"relatedUrls": ["/materials/11?mode=read", null]' in prompt.user_prompt


def test_grounded_prompt_builder_redacts_sensitive_data_before_prompt() -> None:
    builder = GroundedPromptBuilder()
    request = _build_request()
    evidence_result = EvidenceResult(
        intent=ChatIntent.REPORT_LOOKUP,
        basisTime=request.requested_at,
        items=[
            EvidenceItem(
                type="REPORT",
                title="월간 생산 리스크 보고서",
                summary="민감 패턴은 프롬프트 원천 데이터에서 마스킹됩니다.",
                source="reports",
                data={
                    "riskLevel": "WARNING",
                    "apiKey": "sk-abcdefghijklmnopqrstuvwxyz123456",
                    "accessToken": "short-token-value",
                    "notes": [
                        "운영 확인 필요",
                        (
                            "Authorization: Bearer "
                            "abcDEF1234567890abcDEF1234567890abcDEF1234567890"
                        ),
                    ],
                    "nested": {
                        "password": "short-password",
                        "lineCode": "LINE-A01",
                    },
                },
            )
        ],
    )

    prompt = builder.build(
        request,
        evidence_result,
        DocumentSearchResult(sources=[]),
    )

    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in prompt.user_prompt
    assert "short-token-value" not in prompt.user_prompt
    assert "Bearer abcDEF" not in prompt.user_prompt
    assert "short-password" not in prompt.user_prompt
    assert '"apiKey": "[보안 제한]"' in prompt.user_prompt
    assert '"accessToken": "[보안 제한]"' in prompt.user_prompt
    assert '"notes": ["운영 확인 필요", "[보안 제한]"]' in prompt.user_prompt
    assert '"nested": {"lineCode": "LINE-A01", "password": "[보안 제한]"}' in (
        prompt.user_prompt
    )
    assert '"riskLevel": "WARNING"' in prompt.user_prompt


def test_grounded_prompt_builder_limits_sources_and_long_text() -> None:
    builder = GroundedPromptBuilder(
        Settings(
            prompt_max_evidence_items=1,
            prompt_max_document_sources=1,
            prompt_max_summary_chars=20,
            prompt_max_data_chars=25,
        )
    )
    request = _build_request()
    evidence_result = EvidenceResult(
        intent=ChatIntent.REPORT_LOOKUP,
        basisTime=request.requested_at,
        items=[
            EvidenceItem(
                type="ORDER",
                title="첫 번째 RDB 근거",
                summary="A" * 60,
                source="customer_orders",
                data={"longText": "B" * 80},
            ),
            EvidenceItem(
                type="ORDER",
                title="두 번째 RDB 근거",
                summary="제외되어야 하는 근거",
                source="customer_orders",
            ),
        ],
    )
    document_result = DocumentSearchResult(
        sources=[
            ChatSource(
                sourceType="REPORT",
                title="첫 번째 문서 근거",
                summary="C" * 60,
            ),
            ChatSource(
                sourceType="REPORT",
                title="두 번째 문서 근거",
                summary="제외되어야 하는 문서",
            ),
        ]
    )

    prompt = builder.build(request, evidence_result, document_result)

    assert "첫 번째 RDB 근거" in prompt.user_prompt
    assert "두 번째 RDB 근거" not in prompt.user_prompt
    assert "1개 RDB 근거는 프롬프트 길이 제한으로 제외됨" in prompt.user_prompt
    assert "첫 번째 문서 근거" in prompt.user_prompt
    assert "두 번째 문서 근거" not in prompt.user_prompt
    assert "1개 문서 근거는 프롬프트 길이 제한으로 제외됨" in prompt.user_prompt
    assert "AAAAAAAAAAAAAAAAA..." in prompt.user_prompt
    assert "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB" not in prompt.user_prompt
    assert "CCCCCCCCCCCCCCCCC..." in prompt.user_prompt
