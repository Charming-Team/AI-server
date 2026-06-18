from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from app.core.database import engine

PROGRESS_RATIOS = [
    0.03, 0.11, 0.27, 0.06, 0.42, 0.18, 0.34, 0.09,
    0.55, 0.22, 0.15, 0.47, 0.31, 0.08, 0.63, 0.25,
    0.39, 0.12, 0.51, 0.29, 0.17, 0.44, 0.36, 0.21,
    0.58, 0.14, 0.32, 0.07, 0.49, 0.26, 0.41, 0.19,
    0.61, 0.24, 0.37, 0.13, 0.46, 0.32, 0.52, 0.16,
]


def next_id(conn, table_name: str, column_name: str) -> int:
    seq = conn.execute(
        text(f"SELECT pg_get_serial_sequence('{table_name}', '{column_name}')")
    ).scalar()

    if seq:
        return conn.execute(
            text("SELECT nextval(CAST(:seq AS regclass))"),
            {"seq": seq},
        ).scalar_one()

    return conn.execute(
        text(f"SELECT COALESCE(MAX({column_name}), 0) + 1 FROM {table_name}")
    ).scalar_one()


def choose_line_id(conn, product_id: int, preferred_line_id: int | None) -> int:
    if preferred_line_id is not None:
        exists = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1
                FROM production_lines
                WHERE line_id = :lineId
            )
        """), {"lineId": preferred_line_id}).scalar()

        if exists:
            return preferred_line_id

    line_id = conn.execute(text("""
        SELECT plc.line_id
        FROM product_line_capabilities plc
        JOIN production_lines pl
          ON pl.line_id = plc.line_id
        WHERE plc.product_id = :productId
          AND COALESCE(pl.is_active, true) = true
        ORDER BY plc.priority_rank ASC NULLS LAST, plc.line_id ASC
        LIMIT 1
    """), {"productId": product_id}).scalar()

    if line_id is not None:
        return line_id

    line_id = conn.execute(text("""
        SELECT line_id
        FROM production_lines
        WHERE COALESCE(is_active, true) = true
        ORDER BY line_id
        LIMIT 1
    """)).scalar()

    if line_id is None:
        raise RuntimeError("사용 가능한 production_lines row가 없습니다.")

    return line_id


def ensure_plan(conn, row: dict, idx: int) -> dict:
    existing = conn.execute(text("""
        SELECT
            plan_id,
            order_id,
            product_id,
            line_id,
            planned_start_at,
            planned_end_at,
            planned_quantity,
            estimated_duration_hr
        FROM production_plans
        WHERE order_id = :orderId
        ORDER BY
            planned_start_at ASC NULLS LAST,
            plan_sequence ASC NULLS LAST,
            plan_id ASC
        LIMIT 1
    """), {"orderId": row["order_id"]}).mappings().first()

    now = datetime.now(UTC)
    planned_start_at = now - timedelta(hours=2)
    planned_end_at = now + timedelta(hours=8 + idx % 12)

    if existing:
        plan_id = existing["plan_id"]

        conn.execute(text("""
            UPDATE production_plans
            SET
                planned_quantity = :orderQuantity,
                plan_status = CAST('IN_PROGRESS' AS plan_status_enum),
                planned_start_at = COALESCE(planned_start_at, :plannedStartAt),
                planned_end_at = CASE
                    WHEN planned_end_at > COALESCE(planned_start_at, :plannedStartAt)
                    THEN planned_end_at
                    ELSE :plannedEndAt
                END,
                estimated_duration_hr = GREATEST(COALESCE(estimated_duration_hr, 1), 1),
                updated_at = now()
            WHERE plan_id = :planId
        """), {
            "planId": plan_id,
            "orderQuantity": row["order_quantity"],
            "plannedStartAt": planned_start_at,
            "plannedEndAt": planned_end_at,
        })

        return dict(conn.execute(text("""
            SELECT
                plan_id,
                order_id,
                product_id,
                line_id,
                planned_start_at,
                planned_end_at,
                planned_quantity,
                estimated_duration_hr
            FROM production_plans
            WHERE plan_id = :planId
        """), {"planId": plan_id}).mappings().one())

    line_id = choose_line_id(conn, row["product_id"], row["line_id"])
    plan_id = next_id(conn, "production_plans", "plan_id")

    plan_sequence = conn.execute(text("""
        SELECT COALESCE(MAX(plan_sequence), 0) + 1
        FROM production_plans
        WHERE line_id = :lineId
    """), {"lineId": line_id}).scalar_one()

    conn.execute(text("""
        INSERT INTO production_plans (
            plan_id,
            order_id,
            product_id,
            line_id,
            operator_id,
            planned_start_at,
            planned_end_at,
            estimated_duration_hr,
            planned_quantity,
            plan_sequence,
            plan_status,
            created_at,
            updated_at
        )
        VALUES (
            :planId,
            :orderId,
            :productId,
            :lineId,
            NULL,
            :plannedStartAt,
            :plannedEndAt,
            :estimatedDurationHr,
            :plannedQuantity,
            :planSequence,
            CAST('IN_PROGRESS' AS plan_status_enum),
            now(),
            now()
        )
    """), {
        "planId": plan_id,
        "orderId": row["order_id"],
        "productId": row["product_id"],
        "lineId": line_id,
        "plannedStartAt": planned_start_at,
        "plannedEndAt": planned_end_at,
        "estimatedDurationHr": 8 + idx % 12,
        "plannedQuantity": row["order_quantity"],
        "planSequence": plan_sequence,
    })

    return dict(conn.execute(text("""
        SELECT
            plan_id,
            order_id,
            product_id,
            line_id,
            planned_start_at,
            planned_end_at,
            planned_quantity,
            estimated_duration_hr
        FROM production_plans
        WHERE plan_id = :planId
    """), {"planId": plan_id}).mappings().one())


with engine.begin() as conn:
    rows = conn.execute(text("""
        WITH latest AS (
            SELECT DISTINCT ON (apr.order_id)
                apr.prediction_id,
                apr.order_id,
                apr.product_id,
                apr.line_id,
                apr.risk_level::text AS risk_level,
                apr.delay_probability,
                apr.predicted_delay_days,
                apr.model_version
            FROM ai_prediction_results apr
            JOIN customer_orders co
              ON co.order_id = apr.order_id
            WHERE COALESCE(UPPER(co.order_status::text), '') NOT IN ('COMPLETE', 'COMPLETED', 'CANCELLED')
            ORDER BY apr.order_id, apr.prediction_id DESC
        )
        SELECT
            l.prediction_id,
            l.order_id,
            co.order_no,
            co.product_id,
            l.line_id,
            l.risk_level,
            l.delay_probability,
            l.predicted_delay_days,
            co.order_quantity
        FROM latest l
        JOIN customer_orders co
          ON co.order_id = l.order_id
        ORDER BY
            CASE l.risk_level
                WHEN 'CRITICAL' THEN 1
                WHEN 'WARNING' THEN 2
                WHEN 'CAUTION' THEN 3
                WHEN 'SAFE' THEN 4
                ELSE 5
            END,
            l.delay_probability DESC NULLS LAST,
            l.order_id
    """)).mappings().all()

    print("target rows:", len(rows))

    for idx, row in enumerate(rows):
        plan = ensure_plan(conn, dict(row), idx)

        ratio = PROGRESS_RATIOS[idx % len(PROGRESS_RATIOS)]
        if row["risk_level"] == "SAFE":
            ratio = min(ratio, 0.35)

        order_quantity = int(row["order_quantity"] or 0)
        actual_quantity = max(0, min(order_quantity, int(round(order_quantity * ratio))))
        defect_quantity = max(0, int(round(actual_quantity * 0.015)))
        good_quantity = max(actual_quantity - defect_quantity, 0)
        yield_rate = 0 if actual_quantity == 0 else round(good_quantity / actual_quantity, 4)

        existing_result_id = conn.execute(text("""
            SELECT result_id
            FROM production_results
            WHERE plan_id = :planId
        """), {"planId": plan["plan_id"]}).scalar()

        if existing_result_id:
            conn.execute(text("""
                UPDATE production_results
                SET
                    order_id = :orderId,
                    product_id = :productId,
                    line_id = :lineId,
                    planned_start_at = :plannedStartAt,
                    actual_start_at = COALESCE(actual_start_at, :plannedStartAt),
                    planned_end_at = :plannedEndAt,
                    actual_end_at = NULL,
                    planned_quantity = :plannedQuantity,
                    actual_quantity = :actualQuantity,
                    defect_quantity = :defectQuantity,
                    yield_rate = :yieldRate,
                    actual_duration_hr = GREATEST(COALESCE(:estimatedDurationHr, 1), 1),
                    actual_setup_time_hr = 0,
                    actual_delay_hr = 0,
                    is_delayed = false,
                    actual_delay_reason = NULL,
                    result_status = CAST('PARTIAL' AS result_status_enum),
                    updated_at = now()
                WHERE result_id = :resultId
            """), {
                "resultId": existing_result_id,
                "orderId": row["order_id"],
                "productId": plan["product_id"],
                "lineId": plan["line_id"],
                "plannedStartAt": plan["planned_start_at"],
                "plannedEndAt": plan["planned_end_at"],
                "plannedQuantity": order_quantity,
                "actualQuantity": actual_quantity,
                "defectQuantity": defect_quantity,
                "yieldRate": yield_rate,
                "estimatedDurationHr": plan["estimated_duration_hr"],
            })
        else:
            result_id = next_id(conn, "production_results", "result_id")

            conn.execute(text("""
                INSERT INTO production_results (
                    result_id,
                    plan_id,
                    order_id,
                    product_id,
                    line_id,
                    planned_start_at,
                    actual_start_at,
                    planned_end_at,
                    actual_end_at,
                    planned_quantity,
                    actual_quantity,
                    defect_quantity,
                    yield_rate,
                    actual_duration_hr,
                    actual_setup_time_hr,
                    actual_delay_hr,
                    is_delayed,
                    actual_delay_reason,
                    result_status,
                    updated_at
                )
                VALUES (
                    :resultId,
                    :planId,
                    :orderId,
                    :productId,
                    :lineId,
                    :plannedStartAt,
                    :plannedStartAt,
                    :plannedEndAt,
                    NULL,
                    :plannedQuantity,
                    :actualQuantity,
                    :defectQuantity,
                    :yieldRate,
                    GREATEST(COALESCE(:estimatedDurationHr, 1), 1),
                    0,
                    0,
                    false,
                    NULL,
                    CAST('PARTIAL' AS result_status_enum),
                    now()
                )
            """), {
                "resultId": result_id,
                "planId": plan["plan_id"],
                "orderId": row["order_id"],
                "productId": plan["product_id"],
                "lineId": plan["line_id"],
                "plannedStartAt": plan["planned_start_at"],
                "plannedEndAt": plan["planned_end_at"],
                "plannedQuantity": order_quantity,
                "actualQuantity": actual_quantity,
                "defectQuantity": defect_quantity,
                "yieldRate": yield_rate,
                "estimatedDurationHr": plan["estimated_duration_hr"],
            })

    check = conn.execute(text("""
        WITH latest AS (
            SELECT DISTINCT ON (apr.order_id)
                apr.order_id,
                apr.risk_level::text AS risk_level
            FROM ai_prediction_results apr
            JOIN customer_orders co
              ON co.order_id = apr.order_id
            WHERE COALESCE(UPPER(co.order_status::text), '') NOT IN ('COMPLETE', 'COMPLETED', 'CANCELLED')
            ORDER BY apr.order_id, apr.prediction_id DESC
        )
        SELECT
            l.risk_level,
            COUNT(*) AS cnt,
            ROUND(AVG(
                CASE
                    WHEN co.order_quantity > 0
                    THEN LEAST(COALESCE(prsum.completed_quantity, 0), co.order_quantity)::numeric
                         / co.order_quantity::numeric * 100
                    ELSE 0
                END
            ), 1) AS avg_progress
        FROM latest l
        JOIN customer_orders co
          ON co.order_id = l.order_id
        LEFT JOIN (
            SELECT
                order_id,
                SUM(COALESCE(actual_quantity, 0)) AS completed_quantity
            FROM production_results
            GROUP BY order_id
        ) prsum
          ON prsum.order_id = co.order_id
        GROUP BY l.risk_level
        ORDER BY
            CASE l.risk_level
                WHEN 'CRITICAL' THEN 1
                WHEN 'WARNING' THEN 2
                WHEN 'CAUTION' THEN 3
                WHEN 'SAFE' THEN 4
                ELSE 5
            END
    """)).mappings().all()

    print("✅ demo progress seeded")
    for row in check:
        print(dict(row))
