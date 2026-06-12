# app/features/delay_probability/artifact_io.py
"""
Delay probability model artifact loader and single-row inference wrapper.
기존 학습 프로젝트의 artifact_io.py에서 운영 FastAPI 서버에 필요한 부분만 남긴 파일입니다.

- tar.gz artifact 압축 해제
- model/joblib/json artifact 로드
- 단건 predict_one()
- calibrated probability, risk_level, SHAP factor, cause_detail 생성
"""

from __future__ import annotations

import json
import shutil
import tarfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .inference_utils import predict_delay_probability_one


RISK_THRESHOLDS = {
    "SAFE_MAX": 0.10,
    "CAUTION_MAX": 0.40,
    "WARNING_MAX": 0.70,
    "CRITICAL_MAX": 1.00,
}


DEFAULT_MODEL_DIR_NAME = "delay_probability_xgboost_v1.0.0"
DEFAULT_ARTIFACT_TARBALL_NAME = f"{DEFAULT_MODEL_DIR_NAME}.tar.gz"

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
DEFAULT_EXTRACT_ROOT = DEFAULT_ARTIFACTS_DIR / "_extracted"


REQUIRED_ARTIFACT_FILES = [
    "xgb_pipeline.joblib",
    "calibrated_model.joblib",
    "risk_thresholds.json",
    "feature_name_mapping.json",
    "feature_cause_tag_mapping.json",
    "model_metadata.json",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact 파일을 찾을 수 없습니다: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def _tarball_base_name(path: Path) -> str:
    name = path.name
    if name.endswith(".tar.gz"):
        return name[:-7]
    if name.endswith(".tgz"):
        return name[:-4]
    return path.stem


def _validate_artifact_dir(artifact_dir: Path) -> None:
    if not artifact_dir.exists():
        raise FileNotFoundError(f"artifact directory를 찾을 수 없습니다: {artifact_dir}")

    if not artifact_dir.is_dir():
        raise NotADirectoryError(f"artifact path가 directory가 아닙니다: {artifact_dir}")

    missing_files = [
        file_name for file_name in REQUIRED_ARTIFACT_FILES
        if not (artifact_dir / file_name).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "지연 확률 예측 artifact 필수 파일이 누락되었습니다. "
            f"artifact_dir={artifact_dir}, missing_files={missing_files}"
        )


def _safe_extract_tarball(tarball_path: Path, extract_root: Path) -> None:
    """
    path traversal을 방지하면서 tar.gz를 압축 해제합니다.
    """

    extract_root.mkdir(parents=True, exist_ok=True)
    root_resolved = extract_root.resolve()

    with tarfile.open(tarball_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (extract_root / member.name).resolve()
            if not member_path.is_relative_to(root_resolved):
                raise ValueError(
                    "안전하지 않은 tar.gz artifact 경로가 발견되었습니다: "
                    f"{member.name}"
                )

        tar.extractall(extract_root)


def extract_artifact_tarball(
    tarball_path: str | Path,
    *,
    extract_root: str | Path | None = None,
    force: bool = False,
) -> Path:
    """
    delay_probability_xgboost_v1.0.0.tar.gz를 압축 해제하고
    실제 artifact directory 경로를 반환합니다.
    """

    tarball_path = Path(tarball_path)

    if not tarball_path.exists():
        raise FileNotFoundError(f"artifact tar.gz 파일을 찾을 수 없습니다: {tarball_path}")

    if not tarball_path.is_file():
        raise ValueError(f"artifact tar.gz 경로가 파일이 아닙니다: {tarball_path}")

    extract_root_path = Path(extract_root) if extract_root else DEFAULT_EXTRACT_ROOT
    artifact_dir_name = _tarball_base_name(tarball_path)
    expected_artifact_dir = extract_root_path / artifact_dir_name

    if expected_artifact_dir.exists() and not force:
        _validate_artifact_dir(expected_artifact_dir)
        return expected_artifact_dir

    if expected_artifact_dir.exists() and force:
        shutil.rmtree(expected_artifact_dir)

    _safe_extract_tarball(tarball_path, extract_root_path)

    if expected_artifact_dir.exists():
        _validate_artifact_dir(expected_artifact_dir)
        return expected_artifact_dir

    # tar 내부 최상위 폴더명이 예상과 다른 경우를 대비한 fallback입니다.
    candidates = [
        path for path in extract_root_path.iterdir()
        if path.is_dir() and (path / "xgb_pipeline.joblib").exists()
    ]

    if len(candidates) == 1:
        _validate_artifact_dir(candidates[0])
        return candidates[0]

    raise FileNotFoundError(
        "tar.gz 압축 해제 후 artifact directory를 식별할 수 없습니다. "
        f"tarball={tarball_path}, extract_root={extract_root_path}"
    )


def resolve_artifact_dir(
    artifact_path: str | Path | None = None,
    *,
    extract_root: str | Path | None = None,
    force_extract: bool = False,
) -> Path:
    """
    artifact_path가 directory이면 그대로 사용하고,
    tar.gz이면 압축 해제한 directory를 반환합니다.

    artifact_path가 None이면 기본 위치를 사용합니다.
    기본 탐색 순서:
    1. artifacts/delay_probability_xgboost_v1.0.0/
    2. artifacts/delay_probability_xgboost_v1.0.0.tar.gz
    """

    if artifact_path is None:
        default_dir = DEFAULT_ARTIFACTS_DIR / DEFAULT_MODEL_DIR_NAME
        default_tarball = DEFAULT_ARTIFACTS_DIR / DEFAULT_ARTIFACT_TARBALL_NAME

        if default_dir.exists():
            _validate_artifact_dir(default_dir)
            return default_dir

        if default_tarball.exists():
            return extract_artifact_tarball(
                default_tarball,
                extract_root=extract_root,
                force=force_extract,
            )

        raise FileNotFoundError(
            "기본 지연 확률 예측 artifact를 찾을 수 없습니다. "
            f"directory={default_dir}, tarball={default_tarball}"
        )

    path = Path(artifact_path)

    if path.is_dir():
        _validate_artifact_dir(path)
        return path

    if path.is_file() and (path.name.endswith(".tar.gz") or path.name.endswith(".tgz")):
        return extract_artifact_tarball(
            path,
            extract_root=extract_root,
            force=force_extract,
        )

    raise FileNotFoundError(
        "artifact_path는 artifact directory 또는 .tar.gz 파일이어야 합니다. "
        f"입력값: {path}"
    )


def _get_pipeline_step(pipeline: Any, step_name: str) -> Any | None:
    named_steps = getattr(pipeline, "named_steps", None)

    if not named_steps:
        return None

    return named_steps.get(step_name)


def _load_encoded_feature_names(
    preprocessor: Any,
    feature_schema: dict[str, Any],
) -> list[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(map(str, preprocessor.get_feature_names_out()))

    encoded_feature_names = feature_schema.get("encoded_feature_names")
    if encoded_feature_names:
        return list(map(str, encoded_feature_names))

    raise ValueError(
        "encoded feature name을 확인할 수 없습니다. "
        "preprocessor.get_feature_names_out() 또는 feature_schema.encoded_feature_names가 필요합니다."
    )


def to_risk_level(
    prob: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    thresholds = thresholds or RISK_THRESHOLDS

    safe_max = float(thresholds.get("SAFE_MAX", thresholds.get("safe_max", 0.10)))
    caution_max = float(thresholds.get("CAUTION_MAX", thresholds.get("caution_max", 0.40)))
    warning_max = float(thresholds.get("WARNING_MAX", thresholds.get("warning_max", 0.70)))

    if prob <= safe_max:
        return "SAFE"

    if prob <= caution_max:
        return "CAUTION"

    if prob <= warning_max:
        return "WARNING"

    return "CRITICAL"


def normalize_feature_name(feature_name: str) -> str:
    name = str(feature_name).replace("cat__", "").replace("num__", "")

    if name.startswith("product_code_"):
        return "product_code"

    if name.startswith("primary_line_id_"):
        return "primary_line_id"

    if name.startswith("planned_quantity_gap_bin_"):
        return "planned_quantity_gap_bin"

    if name.startswith("duration_to_leadtime_bin_"):
        return "duration_to_leadtime_bin"

    return name


def get_original_feature_value(row: pd.Series, feature: str) -> Any:
    if feature not in row.index:
        return None

    value = row[feature]

    if pd.isna(value):
        return None

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return float(value)

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    return str(value)


def _get_optional_int(row: pd.Series, *column_names: str) -> int | None:
    for column_name in column_names:
        if column_name in row.index and pd.notna(row[column_name]):
            try:
                return int(row[column_name])
            except (TypeError, ValueError):
                return None

    return None


def get_grouped_shap_for_one_row(
    shap_values_1d: np.ndarray,
    encoded_feature_names: list[str] | np.ndarray,
) -> pd.DataFrame:
    temp = pd.DataFrame(
        {
            "encoded_feature": list(encoded_feature_names),
            "normalized_feature": [
                normalize_feature_name(name)
                for name in encoded_feature_names
            ],
            "shap_value": np.asarray(shap_values_1d, dtype=float),
        }
    )

    grouped = (
        temp.groupby("normalized_feature", as_index=False)
        .agg(shap_value=("shap_value", "sum"))
    )

    grouped["abs_shap_value"] = grouped["shap_value"].abs()

    return grouped.sort_values("abs_shap_value", ascending=False)


class DelayProbabilityArtifact:
    """
    FastAPI 서버에서 1회 로드 후 주문 단건 지연 확률 예측에 사용하는 wrapper입니다.
    """

    def __init__(
        self,
        artifact_path: str | Path | None = None,
        *,
        extract_root: str | Path | None = None,
        force_extract: bool = False,
    ) -> None:
        self.artifact_dir = resolve_artifact_dir(
            artifact_path,
            extract_root=extract_root,
            force_extract=force_extract,
        )

        self.xgb_pipeline = joblib.load(self.artifact_dir / "xgb_pipeline.joblib")
        self.calibrated_model = joblib.load(self.artifact_dir / "calibrated_model.joblib")

        self.preprocessor = _get_pipeline_step(self.xgb_pipeline, "preprocess")
        if self.preprocessor is None:
            preprocessor_path = self.artifact_dir / "preprocessor.joblib"
            if not preprocessor_path.exists():
                raise FileNotFoundError(
                    "pipeline.named_steps['preprocess']도 없고 preprocessor.joblib도 없습니다."
                )
            self.preprocessor = joblib.load(preprocessor_path)

        self.xgb_model = _get_pipeline_step(self.xgb_pipeline, "model")
        if self.xgb_model is None:
            xgb_model_path = self.artifact_dir / "xgb_classifier.joblib"
            if not xgb_model_path.exists():
                raise FileNotFoundError(
                    "pipeline.named_steps['model']도 없고 xgb_classifier.joblib도 없습니다."
                )
            self.xgb_model = joblib.load(xgb_model_path)

        self.risk_thresholds = _read_json(self.artifact_dir / "risk_thresholds.json")
        self.feature_name_map = _read_json(self.artifact_dir / "feature_name_mapping.json")
        self.feature_cause_map = _read_json(self.artifact_dir / "feature_cause_tag_mapping.json")
        self.metadata = _read_json(self.artifact_dir / "model_metadata.json")

        feature_schema_path = self.artifact_dir / "feature_schema.json"
        self.feature_schema = (
            _read_json(feature_schema_path)
            if feature_schema_path.exists()
            else {}
        )

        self.encoded_feature_names = _load_encoded_feature_names(
            self.preprocessor,
            self.feature_schema,
        )

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", "xgboost_delay_probability"))

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "v1.0.0"))

    @property
    def probability_output(self) -> str:
        return str(self.metadata.get("probability_output", "calibrated_sigmoid"))

    def predict_one(
        self,
        source_row: dict[str, Any] | pd.Series | pd.DataFrame,
        *,
        top_n: int = 5,
    ) -> dict[str, Any]:
        return predict_delay_probability_one(
            source_row,
            xgb_pipeline=self.xgb_pipeline,
            calibrated_model=self.calibrated_model,
            preprocessor=self.preprocessor,
            xgb_model=self.xgb_model,
            encoded_feature_names=self.encoded_feature_names,
            risk_thresholds=self.risk_thresholds,
            feature_name_map=self.feature_name_map,
            feature_cause_map=self.feature_cause_map,
            metadata=self.metadata,
            top_n=top_n,
        )


def load_delay_probability_artifact(
    artifact_path: str | Path | None = None,
    *,
    extract_root: str | Path | None = None,
    force_extract: bool = False,
) -> DelayProbabilityArtifact:
    return DelayProbabilityArtifact(
        artifact_path=artifact_path,
        extract_root=extract_root,
        force_extract=force_extract,
    )


@lru_cache(maxsize=1)
def get_default_delay_probability_artifact() -> DelayProbabilityArtifact:
    """
    FastAPI process 내에서 기본 artifact를 1회만 로드합니다.
    """

    return DelayProbabilityArtifact()


def clear_default_artifact_cache() -> None:
    """
    테스트 또는 모델 재로딩 시 사용합니다.
    """

    get_default_delay_probability_artifact.cache_clear()