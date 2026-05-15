from uuid import NAMESPACE_URL, uuid5

from app.core.config import Settings
from app.features.chat.document_payload import (
    InternalDocumentInput,
    InternalDocumentPayload,
    QdrantUpsertPoint,
)


class DocumentIndexBuilder:
    def __init__(self, settings: Settings) -> None:
        self.chunk_size = settings.document_chunk_size
        self.chunk_overlap = settings.document_chunk_overlap

    def build_payloads(self, document: InternalDocumentInput) -> list[InternalDocumentPayload]:
        chunks = self._chunk_text(document.content)
        return [
            self._build_payload(document, chunk_text, index, len(chunks))
            for index, chunk_text in enumerate(chunks, start=1)
        ]

    def build_point(
        self,
        payload: InternalDocumentPayload,
        vector: list[float],
    ) -> QdrantUpsertPoint:
        point_id = self._build_point_id(payload)
        return QdrantUpsertPoint(
            id=point_id,
            vector=vector,
            payload=payload,
        )

    def _build_payload(
        self,
        document: InternalDocumentInput,
        chunk_text: str,
        chunk_index: int,
        chunk_count: int,
    ) -> InternalDocumentPayload:
        return InternalDocumentPayload(
            document_id=document.document_id,
            document_type=document.document_type,
            title=document.title,
            chunk_text=chunk_text,
            chunk_id=f"chunk-{chunk_index:04d}",
            summary=document.summary if chunk_count == 1 else None,
            url=document.url,
            reference_type=document.reference_type,
            reference_id=document.reference_id,
            basis_time=document.basis_time,
            allowed_roles=document.allowed_roles,
            departments=document.departments,
            company_name=document.company_name,
            intent_tags=document.intent_tags,
        )

    def _chunk_text(self, text: str) -> list[str]:
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            return []

        chunks: list[str] = []
        current_parts: list[str] = []
        current_length = 0
        for paragraph in normalized_text.split("\n"):
            if len(paragraph) > self.chunk_size:
                self._append_current_chunk(chunks, current_parts)
                current_parts = []
                current_length = 0
                chunks.extend(self._split_long_text(paragraph))
                continue

            next_length = current_length + len(paragraph) + (2 if current_parts else 0)
            if next_length <= self.chunk_size:
                current_parts.append(paragraph)
                current_length = next_length
                continue

            self._append_current_chunk(chunks, current_parts)
            current_parts = [paragraph]
            current_length = len(paragraph)

        self._append_current_chunk(chunks, current_parts)
        return chunks

    def _split_long_text(self, text: str) -> list[str]:
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _append_current_chunk(
        self,
        chunks: list[str],
        current_parts: list[str],
    ) -> None:
        if not current_parts:
            return
        chunks.append("\n\n".join(current_parts).strip())

    def _normalize_text(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)

    def _build_point_id(self, payload: InternalDocumentPayload) -> str:
        chunk_id = payload.chunk_id or "chunk"
        return str(uuid5(NAMESPACE_URL, f"{payload.document_id}:{chunk_id}"))
