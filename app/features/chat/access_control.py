from app.features.chat.schemas import ChatIntent

ADMIN_ROLE = "ADMIN"
OPERATOR_ROLE = "OPERATOR"
EXECUTIVE_ROLE = "EXECUTIVE"
MANUFACTURING_MANAGER_ROLE = "MANUFACTURING_MANAGER"

BUSINESS_ROLES = frozenset(
    {
        OPERATOR_ROLE,
        EXECUTIVE_ROLE,
        MANUFACTURING_MANAGER_ROLE,
    }
)
COMPANY_INFO_INDEXER_ROLES = frozenset(
    {
        ADMIN_ROLE,
        MANUFACTURING_MANAGER_ROLE,
    }
)
QDRANT_DOCUMENT_TYPES = frozenset({"REPORT", "COMPANY_INFO"})

ROLE_INTENT_MATRIX = {
    OPERATOR_ROLE: frozenset(
        {
            ChatIntent.DELIVERY_RISK,
            ChatIntent.MATERIAL_SHORTAGE,
            ChatIntent.PRODUCTION_PLAN,
            ChatIntent.URGENT_ORDER_IMPACT,
            ChatIntent.WORK_PRIORITY,
            ChatIntent.LINE_BOTTLENECK,
            ChatIntent.REPORT_LOOKUP,
        }
    ),
    EXECUTIVE_ROLE: frozenset(
        {
            ChatIntent.DELIVERY_RISK,
            ChatIntent.MATERIAL_SHORTAGE,
            ChatIntent.PRODUCTION_PLAN,
            ChatIntent.URGENT_ORDER_IMPACT,
            ChatIntent.WORK_PRIORITY,
            ChatIntent.LINE_BOTTLENECK,
            ChatIntent.REPORT_LOOKUP,
        }
    ),
    MANUFACTURING_MANAGER_ROLE: frozenset(
        {
            ChatIntent.DELIVERY_RISK,
            ChatIntent.MATERIAL_SHORTAGE,
            ChatIntent.PRODUCTION_PLAN,
            ChatIntent.URGENT_ORDER_IMPACT,
            ChatIntent.WORK_PRIORITY,
            ChatIntent.LINE_BOTTLENECK,
            ChatIntent.REPORT_LOOKUP,
        }
    ),
}

OPERATOR_RESTRICTED_TERMS = (
    "계약 금액",
    "계약금액",
    "패널티",
    "지체상금",
    "매출",
    "수익",
    "손익",
    "비용",
    "원가",
    "금액",
    "돈",
    "financial",
    "finance",
    "contract amount",
    "penalty",
    "revenue",
    "profit",
    "cost",
)
