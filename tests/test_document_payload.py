from datetime import datetime

from app.features.chat.document_payload import InternalDocumentPayload, QdrantSearchPoint


def test_internal_document_payload_strips_required_text() -> None:
    payload = InternalDocumentPayload(
        documentId=" report-202605 ",
        documentType="REPORT",
        title=" 2026년 5월 생산 리스크 보고서 ",
        chunkText="보고서 본문 청크",
    )

    assert payload.document_id == "report-202605"
    assert payload.title == "2026년 5월 생산 리스크 보고서"


def test_internal_document_payload_normalizes_access_metadata() -> None:
    payload = InternalDocumentPayload(
        documentId="report-202605",
        chunkId="chunk-1",
        documentType=" report ",
        title="2026년 5월 생산 리스크 보고서",
        chunkText="보고서 본문 청크",
        allowedRoles=[" executive ", "EXECUTIVE", "manufacturing_manager", " "],
        intentTags=[" report_lookup ", "DELIVERY_RISK", "report_lookup", " "],
    )

    assert payload.document_type == "REPORT"
    assert payload.allowed_roles == ["EXECUTIVE", "MANUFACTURING_MANAGER"]
    assert payload.intent_tags == ["REPORT_LOOKUP", "DELIVERY_RISK"]


def test_internal_document_payload_maps_to_chat_source() -> None:
    payload = InternalDocumentPayload(
        documentId="report-202605",
        chunkId="chunk-1",
        documentType="REPORT",
        title="2026년 5월 생산 리스크 보고서",
        summary="자재 부족과 LINE-A01 병목이 주요 리스크입니다.",
        chunkText="보고서 본문 청크",
        url="/reports/20",
        referenceType="REPORT",
        referenceId=20,
        basisTime=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
        allowedRoles=["EXECUTIVE", "MANUFACTURING_MANAGER"],
        intentTags=["REPORT_LOOKUP", "DELIVERY_RISK"],
    )

    source = payload.to_chat_source()

    assert source.source_type == "REPORT"
    assert source.title == "2026년 5월 생산 리스크 보고서"
    assert source.summary == "자재 부족과 LINE-A01 병목이 주요 리스크입니다."
    assert source.url == "/reports/20"
    assert source.reference_id == 20
    assert source.source == "report-202605:chunk-1"
    assert source.basis_time == datetime.fromisoformat("2026-05-12T10:30:00+09:00")
    assert source.source_origin == "QDRANT"
    assert source.relevance_score is None


def test_qdrant_search_point_maps_payload_to_chat_source() -> None:
    point = QdrantSearchPoint.model_validate(
        {
            "id": "point-1",
            "score": 0.91,
            "payload": {
                "documentId": "material-guide",
                "documentType": "MATERIAL",
                "title": "주요 자재 안전 재고 기준",
                "chunkText": "MAT-001은 안전 재고 200KG 이상을 유지해야 합니다.",
                "url": "/materials",
                "basisTime": "2026-05-12T11:00:00+09:00",
                "allowedRoles": ["MANUFACTURING_MANAGER"],
                "intentTags": ["MATERIAL_SHORTAGE"],
            },
        }
    )

    source = point.to_chat_source()

    assert source.source_type == "MATERIAL"
    assert source.summary == "MAT-001은 안전 재고 200KG 이상을 유지해야 합니다."
    assert source.source == "material-guide"
    assert source.basis_time == datetime.fromisoformat("2026-05-12T11:00:00+09:00")
    assert source.source_origin == "QDRANT"
    assert source.relevance_score == 0.91
