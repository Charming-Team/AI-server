from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from app.features.chat.evidence_aggregation_policy import (
    MATERIAL_SHORTAGE_IMPACT_SUMMARY_TITLE,
    URGENT_ORDER_IMPACT_SUMMARY_TITLE,
)
from app.features.chat.schemas import ChatAnswerRequest, ChatIntent, EvidenceItem
from app.features.chat.source_url_policy import normalize_internal_url


@dataclass(frozen=True)
class NavigationTarget:
    source_type: str
    url: str | None


class NavigationTargetResolver:
    def resolve(
        self,
        item: EvidenceItem,
        *,
        intent: ChatIntent,
        request: ChatAnswerRequest | None = None,
    ) -> NavigationTarget:
        if item.title == MATERIAL_SHORTAGE_IMPACT_SUMMARY_TITLE:
            return NavigationTarget(source_type="PLAN", url="/production-plans?mode=read")
        if item.title == URGENT_ORDER_IMPACT_SUMMARY_TITLE:
            return NavigationTarget(source_type="PLAN", url="/production-plans?mode=read")

        question = request.question if request is not None else ""
        if intent == ChatIntent.MATERIAL_SHORTAGE:
            return self._resolve_material_shortage_target(item, question)
        if intent == ChatIntent.URGENT_ORDER_IMPACT:
            return self._resolve_urgent_order_target(item, question)

        return NavigationTarget(
            source_type=item.type,
            url=normalize_internal_url(item.url),
        )

    def _resolve_material_shortage_target(
        self,
        item: EvidenceItem,
        question: str,
    ) -> NavigationTarget:
        if self._asks_plan_context(question):
            if plan_url := self._plan_url(item):
                return NavigationTarget(source_type="PLAN", url=plan_url)
            if order_no_url := self._plan_filter_url(item):
                return NavigationTarget(source_type="PLAN", url=order_no_url)

        if material_url := self._material_url(item):
            return NavigationTarget(source_type="MATERIAL", url=material_url)

        return NavigationTarget(
            source_type=item.type,
            url=normalize_internal_url(item.url),
        )

    def _resolve_urgent_order_target(
        self,
        item: EvidenceItem,
        question: str,
    ) -> NavigationTarget:
        if self._asks_order_context(question) and not self._asks_plan_context(question):
            if order_url := self._order_url(item):
                return NavigationTarget(source_type="ORDER", url=order_url)
            if order_filter_url := self._order_filter_url(item):
                return NavigationTarget(source_type="ORDER", url=order_filter_url)

        if plan_url := self._plan_url(item):
            return NavigationTarget(source_type="PLAN", url=plan_url)
        if plan_filter_url := self._plan_filter_url(item):
            return NavigationTarget(source_type="PLAN", url=plan_filter_url)

        return NavigationTarget(source_type="PLAN", url="/production-plans?mode=read")

    def _asks_plan_context(self, question: str) -> bool:
        compact_question = self._compact(question)
        return any(
            term in compact_question
            for term in ("생산계획", "계획", "일정", "스케줄", "plan", "schedule")
        )

    def _asks_order_context(self, question: str) -> bool:
        compact_question = self._compact(question)
        return any(term in compact_question for term in ("주문", "오더", "order"))

    def _plan_url(self, item: EvidenceItem) -> str | None:
        plan_id = self._get_data_value(item.data, "planId", "plan_id")
        if plan_id is None:
            return None
        return f"/production-plans/{plan_id}?mode=read"

    def _plan_filter_url(self, item: EvidenceItem) -> str | None:
        order_no = self._get_data_value(item.data, "orderNo", "order_no")
        if not order_no:
            return None
        return f"/production-plans?orderNo={self._quote(order_no)}&mode=read"

    def _order_url(self, item: EvidenceItem) -> str | None:
        order_id = self._get_data_value(item.data, "orderId", "order_id")
        if order_id is None:
            return None
        return f"/orders/{order_id}?mode=read"

    def _order_filter_url(self, item: EvidenceItem) -> str | None:
        order_no = self._get_data_value(item.data, "orderNo", "order_no")
        if not order_no:
            return None
        return f"/orders?orderNo={self._quote(order_no)}&mode=read"

    def _material_url(self, item: EvidenceItem) -> str | None:
        material_id = self._get_data_value(item.data, "materialId", "material_id")
        if material_id is not None:
            return f"/materials/inventory/{material_id}?mode=read"
        material_code = self._get_data_value(item.data, "materialCode", "material_code")
        if material_code:
            return f"/materials/inventories?materialCode={self._quote(material_code)}&mode=read"
        return None

    def _get_data_value(self, data: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in data:
                return data[key]
        return None

    def _quote(self, value: Any) -> str:
        return quote(str(value).strip(), safe="")

    def _compact(self, value: str) -> str:
        return "".join(value.casefold().split()).replace("_", "").replace("-", "")
