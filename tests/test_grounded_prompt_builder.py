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
    assert "섹션 제목을 쓰지 않는다" in prompt.system_prompt
    assert "자연스러운 문단" in prompt.system_prompt
    assert "450~650자 안쪽" in prompt.system_prompt
    assert "원천 JSON, 전체 데이터 덤프" in prompt.system_prompt
    assert "같은 출처를 반복하지 않는다" in prompt.system_prompt
    assert "수치에는 가능한 경우 시간, 일, %, 수량" in prompt.system_prompt
    assert "날짜와 시간은 YYYY.MM.DD HH:mm 형식" in prompt.system_prompt
    assert "내부 이동 경로와 URL은 답변 본문에 쓰지 않는다" in prompt.system_prompt
    assert "현재 상태, 수치, 진행률은 업무 데이터 근거를 우선" in (
        prompt.system_prompt
    )
    assert "사용자 역할:\nEXECUTIVE" in prompt.user_prompt
    assert "역할별 응답 제한:" in prompt.user_prompt
    assert "RDB 근거:\n없음" in prompt.user_prompt
    assert "문서 검색 근거:\n없음" in prompt.user_prompt
    assert "수치, 상태, 날짜는 업무 데이터 근거 또는 문서 근거" in (
        prompt.user_prompt
    )
    assert "총 개수나 몇 개인지 묻는 질문은 업무 데이터 집계 근거" in (
        prompt.user_prompt
    )
    assert "라인 구성, 전체 라인 상태, 가동 중 라인 질문은 업무 데이터 집계 근거" in (
        prompt.user_prompt
    )
    assert "현재 상태 판단은 업무 데이터 근거를 우선" in prompt.user_prompt
    assert "사용자 역할에서 제한되는 내용은 근거에 있어도 답변하지 않는다" in (
        prompt.user_prompt
    )
    assert "권한 제한 안내는 질문이 제한된 정보를 요구하거나 실제로 제한된 경우" in (
        prompt.user_prompt
    )
    assert "전체 답변은 450~650자 안쪽" in prompt.user_prompt
    assert "자연스러운 챗봇 문장" in prompt.user_prompt
    assert "섹션 제목을 본문에 쓰지 않는다" in prompt.user_prompt
    assert "주요 항목을 한 문단으로 이어서 설명" in prompt.user_prompt
    assert "원천 JSON, 전체 데이터 덤프" in prompt.user_prompt
    assert "비율 값은 % 단위" in prompt.user_prompt
    assert "날짜와 시간은 YYYY.MM.DD HH:mm 형식" in prompt.user_prompt
    assert "내부 이동 경로와 URL은 답변 본문에 절대 쓰지 않는다" in (
        prompt.user_prompt
    )
    assert "조회형 질문은 업무 데이터 결과만 중심으로 답하고" in (
        prompt.user_prompt
    )
    assert "RDB, Qdrant, 문서 검색 같은 내부 근거 시스템 이름" in (
        prompt.user_prompt
    )
    assert "추가 질문 유도 문장이나 일반 안내 문장은 쓰지 않는다" in (
        prompt.user_prompt
    )
    assert "확인 필요하다고 자연스럽게 덧붙인다" in prompt.user_prompt


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
    assert "OPERATOR에게는 RDB 근거에서 제공된 권한 허용 정보만 답변한다" in (
        prompt.user_prompt
    )
    assert "Qdrant 문서 검색 근거 중 금액, 계약" in prompt.user_prompt
    assert "질문이 금액성 정보를 요구하지 않으면 권한 제한 안내" in (
        prompt.user_prompt
    )
    assert "금액성을 제외한 회사 개요" in prompt.user_prompt


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
    assert "원천 데이터" not in prompt.user_prompt
    assert '"orderNo": "ORD-202605-001"' not in prompt.user_prompt
    assert "근거 원천: RDB" in prompt.user_prompt
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
    assert "URL: /reports/20?mode=read" not in prompt.user_prompt
    assert "URL: /reports/21?mode=read" not in prompt.user_prompt
    assert "/reports/20?mode=read" not in prompt.user_prompt
    assert "/reports/21?mode=read" not in prompt.user_prompt


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
                summary="데이터 URL은 답변 프롬프트에 그대로 들어가면 안 됩니다.",
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
    assert "원천 데이터" not in prompt.user_prompt
    assert '"reportUrl": "/reports/20?mode=read"' not in prompt.user_prompt
    assert '"detailUrl": "/lines/1?mode=read"' not in prompt.user_prompt
    assert '"relatedUrls": ["/materials/11?mode=read", null]' not in prompt.user_prompt


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
                summary="민감 패턴은 답변 프롬프트에 들어가면 안 됩니다.",
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
    assert "원천 데이터" not in prompt.user_prompt
    assert '"apiKey": "[보안 제한]"' not in prompt.user_prompt
    assert '"accessToken": "[보안 제한]"' not in prompt.user_prompt
    assert '"riskLevel": "WARNING"' not in prompt.user_prompt


def test_grounded_prompt_builder_limits_sources_and_long_text() -> None:
    builder = GroundedPromptBuilder(
        Settings(
            prompt_max_evidence_items=1,
            prompt_max_document_sources=1,
            prompt_max_summary_chars=20,
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


def test_grounded_prompt_builder_limits_total_prompt_length() -> None:
    builder = GroundedPromptBuilder(
        Settings(
            prompt_max_evidence_items=20,
            prompt_max_document_sources=20,
            prompt_max_summary_chars=500,
            prompt_max_total_chars=1000,
        )
    )
    request = _build_request()
    evidence_result = EvidenceResult(
        intent=ChatIntent.REPORT_LOOKUP,
        basisTime=request.requested_at,
        items=[
            EvidenceItem(
                type="REPORT",
                title=f"긴 RDB 근거 {index}",
                summary="A" * 500,
                source="reports",
                data={"detail": "B" * 500},
            )
            for index in range(10)
        ],
    )
    document_result = DocumentSearchResult(
        sources=[
            ChatSource(
                sourceType="COMPANY_INFO",
                title=f"긴 문서 근거 {index}",
                summary="C" * 500,
            )
            for index in range(10)
        ]
    )

    prompt = builder.build(request, evidence_result, document_result)

    assert len(prompt.user_prompt) <= 1000
    assert "프롬프트 전체 길이 제한으로 일부 근거가 생략되었습니다." in (
        prompt.user_prompt
    )
