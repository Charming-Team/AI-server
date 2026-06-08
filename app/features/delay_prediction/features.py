from __future__ import annotations

FEATURE_COLUMNS = [
    "order_quantity",
    "due_gap_days",
    "product_id",
    "average_yield_rate",
    "estimated_duration_hr_sum",
    "plan_overdue_days_max",
    "overdue_plan_ratio",
    "waiting_time_hr_avg",
    "utilization_rate_avg",
]

CATEGORICAL_COLUMNS = ["product_id"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]
TARGET_COLUMN = "actual_delay_hr_sum"
