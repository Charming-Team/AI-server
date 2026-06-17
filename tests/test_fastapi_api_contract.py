from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.features.business_report.schemas.response import BusinessReportGenerateResponse
from app.features.delay_probability.schemas.response import DelayProbabilityPredictResponse
from app.main import create_app


def test_openapi_uses_api_v1_contract_paths() -> None:
    client = TestClient(create_app())
    paths = client.get("/openapi.json").json()["paths"]
    api_prefix = get_settings().api_v1_prefix.rstrip("/")

    assert f"{api_prefix}/chat/answer" in paths
    assert f"{api_prefix}/reports/generate" in paths
    assert f"{api_prefix}/business-reports/generate" in paths
    assert f"{api_prefix}/delay-prediction/predict" in paths
    assert f"{api_prefix}/delay-prediction/health" in paths
    assert f"{api_prefix}/planning" in paths
    assert f"{api_prefix}/health" in paths
    assert f"{api_prefix}/business-report/generate" not in paths
    assert f"{api_prefix}/delay_prediction/predict" not in paths


def test_business_report_response_serializes_external_fields_as_camel_case() -> None:
    response = BusinessReportGenerateResponse(
        report_id=1,
        report_type="PRODUCTION_RISK",
        report_title="월간 리스크 보고서",
        author_id=2,
        target_start_date="2026-06-01",
        target_end_date="2026-06-30",
        report_content={"summary": "ok"},
        created_at="2026-06-17T10:00:00+09:00",
        updated_at="2026-06-17T10:00:00+09:00",
    )

    payload = response.model_dump(mode="json", by_alias=True)

    assert payload["reportId"] == 1
    assert payload["targetStartDate"] == "2026-06-01"
    assert payload["reportContent"] == {"summary": "ok"}
    assert "report_id" not in payload
    assert "target_start_date" not in payload


def test_delay_probability_response_accepts_snake_case_and_serializes_camel_case() -> None:
    response = DelayProbabilityPredictResponse.from_prediction_result(
        {
            "order_id": 314,
            "product_id": 10,
            "plan_id": None,
            "line_id": 5,
            "raw_delay_probability": 0.01,
            "delay_probability": 0.02,
            "risk_level": "SAFE",
            "model_name": "xgboost_delay_probability",
            "model_version": "v1.0.0",
            "probability_output": "calibrated_sigmoid",
            "predicted_at": datetime(2026, 6, 17, tzinfo=UTC),
            "top_factors": [
                {
                    "feature": "capacity_load_ratio",
                    "feature_name_ko": "라인 부하",
                    "cause_tag": "LINE_LOAD",
                    "feature_value": 0.7,
                    "impact": 0.3,
                    "abs_impact": 0.3,
                    "direction": "increase",
                }
            ],
            "risk_increase_factors": [],
            "risk_decrease_factors": [],
            "cause_detail": {
                "raw_delay_probability": 0.01,
                "calibrated_delay_probability": 0.02,
                "probability_output": "calibrated_sigmoid",
                "top_factors": [],
                "risk_increase_factors": [],
                "risk_decrease_factors": [],
            },
        }
    )

    payload = response.model_dump(mode="json", by_alias=True)

    assert payload["orderId"] == 314
    assert payload["delayProbability"] == 0.02
    assert payload["riskLevel"] == "SAFE"
    assert payload["topFactors"][0]["featureNameKo"] == "라인 부하"
    assert payload["topFactors"][0]["absImpact"] == 0.3
    assert payload["causeDetail"]["calibratedDelayProbability"] == 0.02
    assert "order_id" not in payload
    assert "delay_probability" not in payload
    assert "cause_detail" not in payload
