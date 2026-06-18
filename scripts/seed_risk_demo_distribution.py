from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


import json
from typing import Any

from sqlalchemy import text

from app.core.database import engine

MODEL_NAME = "xgboost_delay_probability"
MODEL_VERSION = "v1.0.0-demo-risk-seed"

# true면 production_plans / production_plan_materials도 일부 위험하게 맞춥니다.
# 화면 데모만 목적이면 False로 둬도 충분합니다.
APPLY_SOURCE_DATA_MUTATION = False


def level_from_probability(prob: float) -> str:
    if prob <= 0.1000:
        return "SAFE"
    if prob <= 0.4000:
        return "CAUTION"
    if prob <= 0.7000:
        return "WARNING"
    return "CRITICAL"


def risk_label(level: str) -> str:
    return {
        "CRITICAL": "매우 위험",
        "WARNING": "위험",
        "CAUTION": "주의",
        "SAFE": "안전",
    }[level]


def factor_for_cause(cause_type: str, impact: float) -> dict[str, Any]:
    mapping = {
        "MATERIAL_SHORTAGE": {
            "feature": "planned_quantity_gap_bin",
            "feature_name_ko": "계획 수량 부족 구간",
            "feature_value": "GAP_OVER_10PCT",
            "cause_tag": "MATERIAL_SHORTAGE",
        },
        "MATERIAL_DELAY": {
            "feature": "inbound_delay_days",
            "feature_name_ko": "자재 입고 지연 일수",
            "feature_value": 3.0,
            "cause_tag": "MATERIAL_DELAY",
        },
        "LOW_YIELD": {
            "feature": "avg_standard_yield_rate",
            "feature_name_ko": "제품-라인 평균 표준 수율",
            "feature_value": 0.884,
            "cause_tag": "LOW_YIELD",
        },
        "MACHINE_ABNORMAL": {
            "feature": "machine_status_abnormal",
            "feature_name_ko": "설비 상태 이상",
            "feature_value": "ERROR_OR_MAINTENANCE",
            "cause_tag": "MACHINE_ABNORMAL",
        },
        "LINE_ABNORMAL": {
            "feature": "capacity_load_ratio",
            "feature_name_ko": "라인 생산능력 대비 주문 부하",
            "feature_value": 1.42,
            "cause_tag": "LINE_ABNORMAL",
        },
    }

    base = mapping[cause_type]
    return {
        **base,
        "impact": impact,
        "abs_impact": abs(impact),
        "direction": "increase",
    }


def make_cause_detail(prob: float, cause_types: list[str]) -> str:
    factors = []

    if "MATERIAL_SHORTAGE" in cause_types:
        factors.append(
            {
                "feature": "planned_quantity_gap_bin",
                "feature_name_ko": "계획 수량 부족 구간",
                "cause_tag": "PLAN_QTY_GAP",
                "feature_value": "GAP_OVER_10PCT",
                "impact": 1.42,
                "abs_impact": 1.42,
                "direction": "increase",
            }
        )

    if "MATERIAL_DELAY" in cause_types:
        factors.append(
            {
                "feature": "inbound_delay_days",
                "feature_name_ko": "자재 입고 지연 일수",
                "cause_tag": "MATERIAL_DELAY",
                "feature_value": 3.0,
                "impact": 0.91,
                "abs_impact": 0.91,
                "direction": "increase",
            }
        )

    if "LOW_YIELD" in cause_types:
        factors.append(
            {
                "feature": "avg_standard_yield_rate",
                "feature_name_ko": "제품-라인 평균 표준 수율",
                "cause_tag": "YIELD_RISK",
                "feature_value": 0.884,
                "impact": 0.74,
                "abs_impact": 0.74,
                "direction": "increase",
            }
        )

    if "MACHINE_ABNORMAL" in cause_types:
        factors.append(
            {
                "feature": "machine_status_abnormal",
                "feature_name_ko": "설비 상태 이상",
                "cause_tag": "MACHINE_ABNORMAL",
                "feature_value": "ERROR_OR_MAINTENANCE",
                "impact": 0.88,
                "abs_impact": 0.88,
                "direction": "increase",
            }
        )

    if "LINE_ABNORMAL" in cause_types:
        factors.append(
            {
                "feature": "capacity_load_ratio",
                "feature_name_ko": "라인 생산능력 대비 주문 부하",
                "cause_tag": "LINE_LOAD",
                "feature_value": 1.42,
                "impact": 1.18,
                "abs_impact": 1.18,
                "direction": "increase",
            }
        )

    if prob > 0.7:
        factors.append(
            {
                "feature": "due_margin_to_duration_ratio_capped",
                "feature_name_ko": "생산 소요시간 대비 납기 여유 비율",
                "cause_tag": "DUE_MARGIN_RISK",
                "feature_value": -1.84,
                "impact": 1.33,
                "abs_impact": 1.33,
                "direction": "increase",
            }
        )

    if not factors:
        factors = [
            {
                "feature": "due_margin_to_duration_ratio_capped",
                "feature_name_ko": "생산 소요시간 대비 납기 여유 비율",
                "cause_tag": "DUE_MARGIN_RISK",
                "feature_value": 2.6,
                "impact": -0.2,
                "abs_impact": 0.2,
                "direction": "decrease",
            }
        ]

    payload = {
        "raw_delay_probability": prob,
        "calibrated_delay_probability": prob,
        "probability_output": "demo_seed",
        "top_factors": factors,
        "risk_increase_factors": factors if prob > 0.1 else [],
        "risk_decrease_factors": [] if prob > 0.1 else factors,
    }

    return json.dumps(payload, ensure_ascii=False)


def build_summary(order_no: str, level: str, prob: float, delay_days: float, body: str) -> str:
    return (
        f"{order_no} 주문건은 현재 {risk_label(level)} 단계입니다. "
        f"지연 확률은 {prob * 100:.1f}%, 예상 지연 일수는 {delay_days:.1f}일입니다. "
        f"{body}"
    )


# 24개 non-SAFE를 일부러 섞어서 배정합니다.
# count: CRITICAL 11, WARNING 5, CAUTION 8
risk_slots = [
    {
        "level": "CRITICAL",
        "prob": 0.8756,
        "delay_days": 21.4,
        "causes": ["MATERIAL_SHORTAGE", "LINE_ABNORMAL"],
        "body": (
            "주요 지연 원인은 자재 부족과 라인 상태 이상이 동시에 "
            "발생한 복합 요인입니다. 일부 필요 자재가 부족하며, 배정 "
            "라인의 생산 부하도 높은 상태입니다. 자재가 확보되더라도 "
            "현재 라인 일정이 납기 이후까지 이어질 가능성이 있어 단순 "
            "자재 보충만으로는 납기 리스크를 해소하기 어렵습니다."
        ),
        "action": (
            "부족 자재 확보 가능 시점을 우선 확인하고, 동시에 일부 "
            "물량을 대체 라인으로 분산하십시오. 납기 우선 주문으로 "
            "분류하여 생산계획 수정 화면에서 라인 재배정 또는 계획 "
            "순서 조정을 우선 적용하십시오."
        ),
    },
    {
        "level": "CAUTION",
        "prob": 0.2840,
        "delay_days": 1.3,
        "causes": ["MATERIAL_SHORTAGE"],
        "body": (
            "주요 지연 원인은 자재 부족입니다. 현재 생산계획 기준 "
            "일부 첨가제와 포장재의 예약 수량이 필요 수량보다 부족해 "
            "생산 시작 시점에 전체 투입량을 확보하지 못할 가능성이 "
            "있습니다."
        ),
        "action": (
            "부족 자재의 가용 재고와 입고 예정량을 확인하고, 동일 "
            "자재를 사용하는 후순위 계획의 예약 수량 조정을 "
            "검토하십시오."
        ),
    },
    {
        "level": "WARNING",
        "prob": 0.5270,
        "delay_days": 2.1,
        "causes": ["MATERIAL_DELAY"],
        "body": (
            "주요 지연 원인은 자재 입고 지연입니다. 생산에 필요한 "
            "원료의 입고 예정일이 생산 필요 시점보다 늦어, 현재 "
            "일정대로는 생산 착수가 지연될 가능성이 높습니다."
        ),
        "action": (
            "입고 예정 자재의 조기 입고 가능 여부를 확인하고, 입고 "
            "전까지 동일 라인에 배정 가능한 다른 주문을 선행 배치하는 "
            "방안을 검토하십시오."
        ),
    },
    {
        "level": "CRITICAL",
        "prob": 0.7531,
        "delay_days": 4.6,
        "causes": ["LINE_ABNORMAL"],
        "body": (
            "주요 지연 원인은 라인 상태 이상입니다. 배정 라인의 "
            "가동률이 높고 대기시간이 증가한 상태이며, 현재 주문 수량 "
            "대비 예상 소요시간이 납기 여유를 초과하고 있습니다."
        ),
        "action": (
            "동일 제품 생산이 가능한 대체 라인의 여유 시간을 "
            "확인하고, 납기 우선순위가 낮은 계획을 후순위로 이동하는 "
            "방안을 검토하십시오."
        ),
    },
    {
        "level": "CAUTION",
        "prob": 0.3360,
        "delay_days": 0.8,
        "causes": ["LOW_YIELD"],
        "body": (
            "주요 지연 원인은 수율 저하입니다. 해당 제품-라인 조합의 "
            "기준 수율 대비 예상 수율이 낮아, 동일 주문량을 충족하기 "
            "위한 추가 생산 시간이 필요할 수 있습니다."
        ),
        "action": (
            "해당 제품의 최근 생산 실적과 불량 발생량을 확인하고, "
            "계획 수량에 여유분을 반영하거나 수율이 높은 라인으로 "
            "재배정하는 방안을 검토하십시오."
        ),
    },
    {
        "level": "CRITICAL",
        "prob": 0.8124,
        "delay_days": 2.7,
        "causes": ["MACHINE_ABNORMAL"],
        "body": (
            "주요 지연 원인은 설비 상태 이상입니다. 일부 "
            "Machine의 점검 또는 정지 상태가 확인되어 생산 "
            "흐름이 중단될 가능성이 큽니다."
        ),
        "action": (
            "설비 이상이 있는 Machine의 복구 일정을 확인하고, "
            "동일 제품을 처리할 수 있는 대체 설비 또는 대체 라인 "
            "투입 가능 여부를 검토하십시오."
        ),
    },
    {
        "level": "WARNING",
        "prob": 0.6120,
        "delay_days": 3.4,
        "causes": ["MACHINE_ABNORMAL", "LINE_ABNORMAL"],
        "body": (
            "설비 상태 이상과 라인 부하가 동시에 나타나고 있습니다. "
            "병목 설비가 정상화되지 않으면 후속 공정 대기시간도 증가할 "
            "가능성이 있습니다."
        ),
        "action": (
            "설비 점검 완료 예상 시각을 확인하고, 병목 설비 이후 "
            "공정의 대기 작업량을 줄이도록 생산 순서를 조정하십시오."
        ),
    },
    {
        "level": "CAUTION",
        "prob": 0.1860,
        "delay_days": 0.4,
        "causes": ["LINE_ABNORMAL"],
        "body": (
            "즉시 계획 변경이 필요한 수준은 아니지만, 라인 대기시간이 "
            "증가하고 있어 추적이 필요합니다. 신규 긴급 주문이 "
            "추가되면 지연 위험이 높아질 수 있습니다."
        ),
        "action": (
            "현재 계획은 유지하되 해당 라인의 대기시간과 가동률을 "
            "모니터링하십시오. 신규 긴급 주문 발생 시 납기 여유를 "
            "재계산하십시오."
        ),
    },
    {
        "level": "CRITICAL",
        "prob": 0.7352,
        "delay_days": 8.3,
        "causes": ["MATERIAL_SHORTAGE", "LINE_ABNORMAL"],
        "body": (
            "자재 부족과 라인 부하가 동시에 나타나 현재 생산계획 유지 "
            "시 납기 내 완료 가능성이 낮습니다."
        ),
        "action": (
            "자재 확보 일정과 대체 라인 투입 가능성을 동시에 "
            "확인하고, 분할 생산 또는 라인 재배정을 검토하십시오."
        ),
    },
    {
        "level": "CAUTION",
        "prob": 0.2270,
        "delay_days": 1.1,
        "causes": ["MATERIAL_DELAY"],
        "body": (
            "주요 지연 원인은 자재 입고 지연입니다. 자재 수량은 확보 "
            "가능하지만 투입 가능 시점이 늦어지는 것이 핵심 "
            "리스크입니다."
        ),
        "action": (
            "입고 예정 자재의 조기 입고 가능 여부를 확인하고, 입고 "
            "전까지 선행 가능한 계획을 우선 배치하십시오."
        ),
    },
    {
        "level": "CRITICAL",
        "prob": 0.9135,
        "delay_days": 0.9,
        "causes": ["MATERIAL_DELAY", "LINE_ABNORMAL"],
        "body": (
            "예상 지연 일수는 크지 않지만, 자재 투입 시점과 라인 "
            "부하가 동시에 불안정해 지연 발생 확률이 매우 높습니다."
        ),
        "action": (
            "자재 투입 가능 시각을 확정하고, 같은 라인에 배정된 후속 "
            "계획의 순서를 조정해 짧은 지연이 확산되지 않도록 "
            "관리하십시오."
        ),
    },
    {
        "level": "WARNING",
        "prob": 0.4890,
        "delay_days": 1.7,
        "causes": ["LOW_YIELD", "MACHINE_ABNORMAL"],
        "body": (
            "주요 지연 원인은 수율 저하와 설비 상태 이상입니다. 최근 "
            "동일 제품 생산에서 수율이 기준보다 낮고, 일부 설비 "
            "상태도 정상 가동 상태가 아닙니다."
        ),
        "action": (
            "설비 복구 일정을 확인한 뒤 예상 수율을 반영하여 계획 "
            "수량을 보정하십시오. 필요 시 안정 라인으로 일부 물량 "
            "이전을 검토하십시오."
        ),
    },
    {
        "level": "CRITICAL",
        "prob": 0.7818,
        "delay_days": 11.2,
        "causes": ["LINE_ABNORMAL"],
        "body": (
            "배정 라인의 작업량이 집중되어 납기 여유 대비 생산 "
            "소요시간이 과도합니다. 현재 계획을 유지하면 납기 내 완료 "
            "가능성이 낮습니다."
        ),
        "action": (
            "생산계획 수정 화면에서 해당 주문의 라인 재배정 가능 "
            "여부를 확인하고, 후순위 주문의 시작 시각 조정을 "
            "검토하십시오."
        ),
    },
    {
        "level": "CAUTION",
        "prob": 0.3190,
        "delay_days": 2.0,
        "causes": ["LOW_YIELD"],
        "body": (
            "수율 저하로 인해 주문 수량 충족을 위한 추가 생산 시간이 "
            "필요할 수 있습니다. 납기 여유가 충분하지 않으므로 추적이 "
            "필요합니다."
        ),
        "action": (
            "예상 불량량을 반영하여 계획 수량을 보정하고, 수율이 높은 "
            "라인으로 일부 물량 이동을 검토하십시오."
        ),
    },
    {
        "level": "WARNING",
        "prob": 0.5810,
        "delay_days": 2.8,
        "causes": ["LINE_ABNORMAL"],
        "body": (
            "납기 여유 대비 생산 소요시간이 높고 배정 라인의 가동률이 "
            "높아 생산 흐름 지연 가능성이 있습니다."
        ),
        "action": "라인 재배정 또는 후순위 주문의 시작 시각 조정을 검토하십시오.",
    },
    {
        "level": "CRITICAL",
        "prob": 0.8467,
        "delay_days": 1.4,
        "causes": ["MACHINE_ABNORMAL"],
        "body": (
            "설비 이상으로 인한 생산 흐름 중단 가능성이 높습니다. "
            "예상 지연 일수는 제한적이지만 지연 발생 확률이 높아 즉시 "
            "확인이 필요합니다."
        ),
        "action": (
            "설비 담당자에게 복구 예상 시간을 확인하고, 해당 공정의 "
            "우회 생산 가능 여부를 검토하십시오."
        ),
    },
    {
        "level": "CAUTION",
        "prob": 0.1280,
        "delay_days": 0.2,
        "causes": ["LINE_ABNORMAL"],
        "body": (
            "라인 대기시간 증가가 관찰되지만 현재는 납기 여유가 남아 "
            "있습니다. 다만 후속 주문 추가 시 위험이 상승할 수 "
            "있습니다."
        ),
        "action": "현재 계획은 유지하되 라인 대기시간과 가동률을 모니터링하십시오.",
    },
    {
        "level": "CRITICAL",
        "prob": 0.7042,
        "delay_days": 6.8,
        "causes": ["MATERIAL_SHORTAGE"],
        "body": "필요 자재의 예약 수량이 부족해 생산 착수 자체가 지연될 가능성이 큽니다.",
        "action": (
            "부족 자재의 대체 조달 가능성을 확인하고, 확보 시점에 "
            "맞춰 생산 시작 시각을 재조정하십시오."
        ),
    },
    {
        "level": "WARNING",
        "prob": 0.6470,
        "delay_days": 0.9,
        "causes": ["MATERIAL_DELAY", "LOW_YIELD"],
        "body": (
            "자재 투입 시점 지연과 예상 수율 저하가 함께 나타나 생산 "
            "수량 충족에 불확실성이 있습니다."
        ),
        "action": (
            "입고 일정과 예상 수율을 동시에 반영하여 계획 수량과 시작 시각을 재검토하십시오."
        ),
    },
    {
        "level": "CAUTION",
        "prob": 0.3650,
        "delay_days": 1.6,
        "causes": ["MACHINE_ABNORMAL"],
        "body": (
            "일부 설비 상태 이상이 생산 흐름에 영향을 줄 수 "
            "있습니다. 현재는 즉시 지연 확정 단계는 아니지만 점검이 "
            "필요합니다."
        ),
        "action": (
            "설비 상태를 재확인하고, 이상 지속 시 안정 라인으로 일부 물량 이전을 검토하십시오."
        ),
    },
    {
        "level": "CRITICAL",
        "prob": 0.8891,
        "delay_days": 3.2,
        "causes": ["LINE_ABNORMAL", "LOW_YIELD"],
        "body": (
            "라인 부하와 수율 저하가 동시에 나타나 현재 계획 유지 시 "
            "납기 내 완료 가능성이 매우 낮습니다."
        ),
        "action": "라인 부하를 분산하고 수율 보정 수량을 계획에 반영하십시오.",
    },
    {
        "level": "CAUTION",
        "prob": 0.1520,
        "delay_days": 0.5,
        "causes": ["MATERIAL_DELAY"],
        "body": (
            "자재 입고 일정이 생산 필요 시점과 가까워 추적이 "
            "필요합니다. 입고가 하루만 늦어져도 생산 착수가 지연될 수 "
            "있습니다."
        ),
        "action": ("자재 입고 확정 시간을 확인하고, 입고 전 선행 가능한 작업을 우선 배치하십시오."),
    },
    {
        "level": "CRITICAL",
        "prob": 0.7629,
        "delay_days": 18.5,
        "causes": ["MATERIAL_SHORTAGE", "MACHINE_ABNORMAL"],
        "body": (
            "자재 부족과 설비 상태 이상이 동시에 발생해 현재 생산계획 "
            "유지 시 장기 지연 가능성이 높습니다."
        ),
        "action": (
            "자재 확보와 설비 복구 계획을 동시에 수립하고, 생산계획 "
            "수정 화면에서 대체 라인 분산안을 우선 적용하십시오."
        ),
    },
    {
        "level": "CRITICAL",
        "prob": 0.9284,
        "delay_days": 0.6,
        "causes": ["LINE_ABNORMAL"],
        "body": (
            "예상 지연 일수는 짧지만 라인 부하와 순서 제약으로 인해 "
            "지연 발생 확률이 매우 높습니다. 단기 대응을 놓치면 후속 "
            "주문까지 영향이 확산될 수 있습니다."
        ),
        "action": (
            "해당 주문을 납기 우선순위로 올리고, 같은 라인의 후속 계획 순서를 즉시 조정하십시오."
        ),
    },
]


def safe_slot(idx: int) -> dict[str, Any]:
    probs = [0.015, 0.022, 0.031, 0.038, 0.044, 0.052, 0.066, 0.081]
    return {
        "level": "SAFE",
        "prob": probs[idx % len(probs)],
        "delay_days": 0.0,
        "causes": [],
        "body": "",
        "action": None,
    }


def allocate_prediction_ids(conn, count: int) -> list[int]:
    sequence_name = conn.execute(
        text("SELECT pg_get_serial_sequence('ai_prediction_results', 'prediction_id')")
    ).scalar()

    if sequence_name:
        return [
            conn.execute(
                text("SELECT nextval(CAST(:seq AS regclass))"),
                {"seq": sequence_name},
            ).scalar_one()
            for _ in range(count)
        ]

    base = conn.execute(
        text("SELECT COALESCE(MAX(prediction_id), 0) FROM ai_prediction_results")
    ).scalar_one()

    return [base + idx + 1 for idx in range(count)]


def apply_source_mutation(conn, row: dict[str, Any], slot: dict[str, Any]) -> None:
    """
    데모 일관성 보강용입니다.
    - 매우위험/위험/주의 주문의 production_plans를 납기 근처 또는 납기 이후로 밀어 둡니다.
    - 자재 원인이 있으면 해당 plan의 production_plan_materials를 shortage 상태로 둡니다.
    """
    if slot["level"] == "SAFE":
        return

    risk_level = slot["level"]

    if row.get("plan_id") is not None:
        # 계획 종료를 납기 이후로 맞춰 위험 데이터처럼 보이게 합니다.
        # 단, 실제 UI의 생산 진행률은 production_results 기준이라
        # 이 업데이트만으로 진행률이 바뀌지는 않습니다.
        conn.execute(
            text("""
                UPDATE production_plans pp
                SET
                    planned_end_at = (
                        (co.due_date::timestamp + (:delayDays || ' days')::interval)
                    )::timestamptz,
                    estimated_duration_hr = GREATEST(
                        pp.estimated_duration_hr,
                        (CAST(:delayDays AS numeric) * 24.0) + 8.0
                    ),
                    plan_status = CASE
                        WHEN :riskLevel = 'CRITICAL' THEN CAST('DELAYED' AS plan_status_enum)
                        ELSE pp.plan_status
                    END,
                    updated_at = now()
                FROM customer_orders co
                WHERE pp.order_id = co.order_id
                  AND pp.plan_id = :planId
                  AND (
                        (co.due_date::timestamp + (:delayDays || ' days')::interval)::timestamptz
                        > pp.planned_start_at
                  )
            """),
            {
                "planId": row["plan_id"],
                "delayDays": str(max(float(slot["delay_days"]), 0.1)),
                "riskLevel": risk_level,
            },
        )

    if row.get("plan_id") is not None and (
        "MATERIAL_SHORTAGE" in slot["causes"] or "MATERIAL_DELAY" in slot["causes"]
    ):
        conn.execute(
            text("""
                UPDATE production_plan_materials ppm
                SET
                    reserved_quantity = 0,
                    shortage_quantity = required_quantity,
                    material_plan_status = CAST('SHORTAGE' AS material_plan_status_enum),
                    updated_at = now()
                WHERE ppm.plan_id = :planId
                  AND ppm.required_quantity > 0
            """),
            {"planId": row["plan_id"]},
        )


def main() -> None:
    with engine.begin() as conn:
        eligible_rows = (
            conn.execute(
                text("""
                SELECT
                    co.order_id,
                    co.order_no,
                    co.product_id,
                    co.order_quantity,
                    co.due_date,
                    p.product_name,
                    pp.plan_id,
                    pp.line_id,
                    pl.line_name
                FROM customer_orders co
                JOIN products p
                  ON p.product_id = co.product_id
                LEFT JOIN LATERAL (
                    SELECT
                        pp.plan_id,
                        pp.line_id
                    FROM production_plans pp
                    WHERE pp.order_id = co.order_id
                    ORDER BY
                        pp.planned_start_at ASC NULLS LAST,
                        pp.plan_sequence ASC NULLS LAST,
                        pp.plan_id ASC
                    LIMIT 1
                ) pp ON true
                LEFT JOIN production_lines pl
                  ON pl.line_id = pp.line_id
                WHERE COALESCE(UPPER(co.order_status::text), '') NOT IN (
                        'COMPLETE',
                        'COMPLETED',
                        'CANCELLED'
                  )
                  AND EXISTS (
                        SELECT 1
                        FROM delay_prediction_evidence.vw_delay_probability_inference_orders v
                        WHERE v.order_id = co.order_id
                  )
                ORDER BY co.due_date ASC, co.order_id ASC
            """)
            )
            .mappings()
            .all()
        )

        if len(eligible_rows) < len(risk_slots):
            raise RuntimeError(
                "데모 seed 대상 미완료 주문이 부족합니다. "
                f"필요={len(risk_slots)}, 현재={len(eligible_rows)}"
            )

        # 기존 데모 seed 제거. 원래 모델 row는 건드리지 않습니다.
        old_demo_ids = (
            conn.execute(
                text("""
                SELECT prediction_id
                FROM ai_prediction_results
                WHERE model_version = :modelVersion
            """),
                {"modelVersion": MODEL_VERSION},
            )
            .scalars()
            .all()
        )

        if old_demo_ids:
            conn.execute(
                text("DELETE FROM ai_prediction_causes WHERE prediction_id = ANY(:ids)"),
                {"ids": list(old_demo_ids)},
            )
            conn.execute(
                text("DELETE FROM ai_prediction_results WHERE prediction_id = ANY(:ids)"),
                {"ids": list(old_demo_ids)},
            )

        prediction_ids = allocate_prediction_ids(conn, len(eligible_rows))

        for idx, row in enumerate(eligible_rows):
            slot = risk_slots[idx] if idx < len(risk_slots) else safe_slot(idx)

            prob = float(slot["prob"])
            level = level_from_probability(prob)

            if level != slot["level"]:
                raise RuntimeError(
                    f"slot level mismatch. prob={prob}, expected={slot['level']}, actual={level}"
                )
            delay_days = float(slot["delay_days"])
            cause_types = list(slot["causes"])

            if level != "SAFE" and not cause_types:
                cause_types = ["LINE_ABNORMAL"]
            analysis_summary = None
            recommended_action = None

            if level != "SAFE":
                analysis_summary = build_summary(
                    row["order_no"],
                    level,
                    prob,
                    delay_days,
                    slot["body"],
                )
                recommended_action = slot["action"]

            cause_detail = make_cause_detail(prob, cause_types)

            if APPLY_SOURCE_DATA_MUTATION:
                apply_source_mutation(conn, row, slot)

            prediction_id = prediction_ids[idx]

            conn.execute(
                text("""
                    INSERT INTO ai_prediction_results (
                        prediction_id,
                        order_id,
                        plan_id,
                        product_id,
                        line_id,
                        delay_probability,
                        risk_level,
                        predicted_delay_days,
                        cause_detail,
                        analysis_summary,
                        recommended_action,
                        model_name,
                        model_version,
                        is_notified,
                        is_checked,
                        predicted_at
                    )
                    VALUES (
                        :predictionId,
                        :orderId,
                        :planId,
                        :productId,
                        :lineId,
                        :delayProbability,
                        CAST(:riskLevel AS risk_level_enum),
                        :predictedDelayDays,
                        CAST(:causeDetail AS jsonb),
                        :analysisSummary,
                        :recommendedAction,
                        :modelName,
                        :modelVersion,
                        false,
                        false,
                        now()
                    )
                """),
                {
                    "predictionId": prediction_id,
                    "orderId": row["order_id"],
                    "planId": row["plan_id"],
                    "productId": row["product_id"],
                    "lineId": row["line_id"],
                    "delayProbability": prob,
                    "riskLevel": level,
                    "predictedDelayDays": delay_days,
                    "causeDetail": cause_detail,
                    "analysisSummary": analysis_summary,
                    "recommendedAction": recommended_action,
                    "modelName": MODEL_NAME,
                    "modelVersion": MODEL_VERSION,
                },
            )

            if cause_types:
                for cause_type in sorted(set(cause_types)):
                    conn.execute(
                        text("""
                            INSERT INTO ai_prediction_causes (
                                prediction_id,
                                cause_type
                            )
                            VALUES (
                                :predictionId,
                                CAST(:causeType AS delay_cause_type_enum)
                            )
                            ON CONFLICT DO NOTHING
                        """),
                        {
                            "predictionId": prediction_id,
                            "causeType": cause_type,
                        },
                    )

        result = (
            conn.execute(
                text("""
                WITH latest AS (
                    SELECT DISTINCT ON (apr.order_id)
                        apr.order_id,
                        apr.risk_level::text AS risk_level,
                        apr.delay_probability,
                        apr.predicted_delay_days,
                        apr.model_version
                    FROM ai_prediction_results apr
                    JOIN customer_orders co
                      ON co.order_id = apr.order_id
                    WHERE COALESCE(UPPER(co.order_status::text), '') NOT IN (
                        'COMPLETE',
                        'COMPLETED',
                        'CANCELLED'
                  )
                    ORDER BY apr.order_id, apr.prediction_id DESC
                )
                SELECT
                    risk_level,
                    COUNT(*) AS count,
                    MIN(delay_probability) AS min_probability,
                    MAX(delay_probability) AS max_probability
                FROM latest
                GROUP BY risk_level
                ORDER BY
                    CASE risk_level
                        WHEN 'CRITICAL' THEN 1
                        WHEN 'WARNING' THEN 2
                        WHEN 'CAUTION' THEN 3
                        WHEN 'SAFE' THEN 4
                        ELSE 5
                    END
            """)
            )
            .mappings()
            .all()
        )

        print("✅ demo risk distribution seeded")
        for item in result:
            print(dict(item))


if __name__ == "__main__":
    main()
