from app.features.chat.schemas import DocumentSearchResult, EvidenceResult


class GroundedFallbackAnswerBuilder:
    _max_items = 3
    _max_summary_chars = 140
    _aggregate_summary_titles = frozenset(
        {
            "공정 라인 전체 현황",
            "생산 라인 구성 전체 현황",
            "가동 라인 전체 현황",
            "긴급 주문 전체 생산계획 영향",
            "자재 부족 영향 생산계획",
        }
    )
    _aggregate_action_sentences = {
        "긴급 주문 전체 생산계획 영향": "상세 일정은 생산계획 화면에서 확인할 수 있습니다.",
        "자재 부족 영향 생산계획": (
            "영향받는 작업의 상세 일정은 생산계획 화면에서 확인할 수 있습니다."
        ),
        "공정 라인 전체 현황": "라인별 상세 상태는 생산 라인 화면에서 확인할 수 있습니다.",
        "생산 라인 구성 전체 현황": "라인별 상세 상태는 생산 라인 화면에서 확인할 수 있습니다.",
        "가동 라인 전체 현황": "라인별 상세 상태는 생산 라인 화면에서 확인할 수 있습니다.",
    }

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
            return "확인된 업무 데이터와 문서를 기준으로 답변합니다."
        if has_rdb_evidence:
            return "확인된 업무 데이터를 기준으로 답변합니다."
        if has_document_sources:
            return "확인된 문서를 기준으로 답변합니다."
        return "확인된 내부 근거를 기준으로 답변합니다."

    def _build_source_sentence(
        self,
        source_label: str,
        items: list[tuple[str, str]],
    ) -> str:
        if not items:
            return ""
        if len(items) == 1 and items[0][0] in self._aggregate_summary_titles:
            return self._build_aggregate_sentence(items[0][0], items[0][1])

        source_fragments = " ".join(
            f"{item_title}에서는 {self._normalize_sentence(item_summary)}"
            for item_title, item_summary in items
        )
        if source_label == "문서":
            return f"참고 문서에는 {source_fragments}"
        return f"주요 확인 내용은 {source_fragments}"

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

    def _normalize_aggregate_sentence(self, text: str) -> str:
        normalized_text = text.strip()
        if normalized_text.endswith((".", "?", "!", "다.", "요.")):
            return normalized_text
        return f"{normalized_text}."

    def _build_aggregate_sentence(self, title: str, summary: str) -> str:
        sentence = self._normalize_aggregate_sentence(summary)
        action_sentence = self._aggregate_action_sentences.get(title)
        if action_sentence and action_sentence not in sentence:
            return f"{sentence} {action_sentence}"
        return sentence
