from app.features.chat.schemas import DocumentSearchResult, EvidenceResult


class GroundedFallbackAnswerBuilder:
    _max_items = 3
    _max_summary_chars = 140

    def build(
        self,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> str:
        sentences = [self._build_intro(evidence_result, document_result)]

        if evidence_result.items:
            sentences.append(
                self._build_source_sentence(
                    "RDB",
                    self._select_items(
                        [(item.title, item.summary) for item in evidence_result.items],
                    ),
                )
            )

        if document_result.sources:
            sentences.append(
                self._build_source_sentence(
                    "문서",
                    self._select_items(
                        [
                            (source.title, source.summary)
                            for source in document_result.sources
                        ],
                    ),
                )
            )

        sentences.append("위 근거에 없는 세부 원인이나 조치는 추가 확인이 필요합니다.")
        return " ".join(sentence for sentence in sentences if sentence)

    def _build_intro(
        self,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> str:
        has_rdb_evidence = bool(evidence_result.items)
        has_document_sources = bool(document_result.sources)
        if has_rdb_evidence and has_document_sources:
            return "확인된 RDB 근거와 문서 검색 근거 기준으로 요약합니다."
        if has_rdb_evidence:
            return "확인된 RDB 근거 기준으로 요약합니다."
        if has_document_sources:
            return "확인된 문서 검색 근거 기준으로 요약합니다."
        return "확인된 내부 근거 기준으로 요약합니다."

    def _build_source_sentence(
        self,
        source_label: str,
        items: list[tuple[str, str]],
    ) -> str:
        if not items:
            return ""

        source_fragments = " ".join(
            f"{item_title}에서는 {self._normalize_sentence(item_summary)}"
            for item_title, item_summary in items
        )
        return f"{source_label} 근거로는 {source_fragments}"

    def _select_items(self, items: list[tuple[str, str]]) -> list[tuple[str, str]]:
        selected_items: list[tuple[str, str]] = []
        seen_titles: set[str] = set()
        for item_title, item_summary in items:
            normalized_title = item_title.strip().casefold()
            if normalized_title in seen_titles:
                continue
            seen_titles.add(normalized_title)
            selected_items.append((item_title, item_summary))
            if len(selected_items) >= self._max_items:
                break
        return selected_items

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_summary_chars:
            return text
        return f"{text[: self._max_summary_chars - 3]}..."

    def _normalize_sentence(self, text: str) -> str:
        truncated_text = self._truncate(text).strip()
        if truncated_text.endswith((".", "?", "!", "다.", "요.")):
            return truncated_text
        return f"{truncated_text}."
