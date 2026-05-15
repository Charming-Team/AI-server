from app.features.chat.document_payload import InternalDocumentPayload, QdrantSearchPoint


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
                "allowedRoles": ["MANUFACTURING_MANAGER"],
                "intentTags": ["MATERIAL_SHORTAGE"],
            },
        }
    )

    source = point.to_chat_source()

    assert source.source_type == "MATERIAL"
    assert source.summary == "MAT-001은 안전 재고 200KG 이상을 유지해야 합니다."
    assert source.source == "material-guide"
