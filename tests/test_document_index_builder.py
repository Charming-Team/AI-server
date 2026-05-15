from datetime import datetime

from app.core.config import Settings
from app.features.chat.document_index_builder import DocumentIndexBuilder
from app.features.chat.document_payload import InternalDocumentInput


def test_document_index_builder_builds_single_payload_with_metadata() -> None:
    builder = DocumentIndexBuilder(Settings(document_chunk_size=500))
    document = InternalDocumentInput(
        documentId=" report-202605 ",
        documentType="REPORT",
        title=" 2026년 5월 생산 리스크 보고서 ",
        content="자재 부족과 LINE-A01 병목이 주요 리스크입니다.",
        summary="월간 생산 리스크 요약",
        url="/reports/20",
        referenceType="REPORT",
        referenceId=20,
        basisTime=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
        allowedRoles=["EXECUTIVE", "MANUFACTURING_MANAGER"],
        companyName="S-MAP",
        intentTags=["REPORT_LOOKUP", "DELIVERY_RISK"],
    )

    payloads = builder.build_payloads(document)

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload.document_id == "report-202605"
    assert payload.title == "2026년 5월 생산 리스크 보고서"
    assert payload.chunk_id == "chunk-0001"
    assert payload.chunk_text == "자재 부족과 LINE-A01 병목이 주요 리스크입니다."
    assert payload.summary == "월간 생산 리스크 요약"
    assert payload.allowed_roles == ["EXECUTIVE", "MANUFACTURING_MANAGER"]
    assert payload.intent_tags == ["REPORT_LOOKUP", "DELIVERY_RISK"]


def test_document_index_builder_splits_long_document_with_overlap() -> None:
    builder = DocumentIndexBuilder(
        Settings(document_chunk_size=20, document_chunk_overlap=5)
    )
    document = InternalDocumentInput(
        documentId="process-guide",
        documentType="PROCESS",
        title="라인 병목 대응 가이드",
        content="A" * 55,
        allowedRoles=["MANUFACTURING_MANAGER"],
        intentTags=["LINE_BOTTLENECK"],
    )

    payloads = builder.build_payloads(document)

    assert len(payloads) == 4
    assert all(len(payload.chunk_text) <= 20 for payload in payloads)
    assert payloads[0].chunk_id == "chunk-0001"
    assert payloads[-1].chunk_id == "chunk-0004"
    assert payloads[1].chunk_text.startswith("A" * 5)


def test_document_index_builder_builds_qdrant_upsert_point() -> None:
    builder = DocumentIndexBuilder(Settings(document_chunk_size=500))
    document = InternalDocumentInput(
        documentId="material-guide",
        documentType="MATERIAL",
        title="자재 안전 재고 기준",
        content="MAT-001은 안전 재고 200KG 이상을 유지해야 합니다.",
        allowedRoles=["MANUFACTURING_MANAGER"],
        intentTags=["MATERIAL_SHORTAGE"],
    )
    payload = builder.build_payloads(document)[0]

    point = builder.build_point(payload, [0.1, 0.2, 0.3])
    qdrant_point = point.to_qdrant_point()

    assert qdrant_point["id"] == point.id
    assert qdrant_point["vector"] == [0.1, 0.2, 0.3]
    assert qdrant_point["payload"]["documentId"] == "material-guide"
    assert qdrant_point["payload"]["chunkId"] == "chunk-0001"
