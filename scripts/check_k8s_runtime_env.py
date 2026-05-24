import argparse
import json
import sys
from pathlib import Path
from typing import TextIO
from urllib.parse import urlparse

K8S_ENV_EXAMPLE_PATH = "deploy/kubernetes/fastapi-chat-runtime.env.example"
PLACEHOLDER_PREFIX = "__SET_BY_SECRET"
CONFIG_PLACEHOLDER_PREFIX = "__SET_"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

REQUIRED_EXACT_VALUES = {
    "ENVIRONMENT": "kubernetes",
    "API_V1_PREFIX": "/ai/api/v1",
    "RDB_EVIDENCE_ENABLED": "true",
    "QDRANT_SEARCH_ENABLED": "true",
    "QDRANT_COLLECTION": "smap_internal_documents",
    "EMBEDDING_ENABLED": "true",
    "EMBEDDING_PATH": "/embed",
    "EMBEDDING_MODEL": "BAAI/bge-m3",
    "EMBEDDING_DIMENSION": "1024",
    "LLM_ENABLED": "true",
    "LLM_PROVIDER": "openai",
    "LLM_BASE_URL": "https://api.openai.com/v1",
    "LLM_MAX_TOKENS": "512",
    "LLM_RESPONSE_CACHE_ENABLED": "true",
    "LLM_RESPONSE_CACHE_TTL_SECONDS": "60.0",
    "LLM_RESPONSE_CACHE_MAX_ENTRIES": "128",
    "PROMPT_MAX_TOTAL_CHARS": "6000",
}

REQUIRED_SECRET_KEYS = {
    "CHAT_ANSWER_INTERNAL_TOKEN",
    "CHAT_RECOMMENDATION_INTERNAL_TOKEN",
    "DOCUMENT_INDEX_INTERNAL_TOKEN",
    "LLM_API_KEY",
    "RDB_EVIDENCE_DSN",
}

REQUIRED_CONFIG_KEYS = {
    "LLM_ALLOWED_MODELS",
    "LLM_MODEL",
}

K8S_SERVICE_URL_KEYS = {
    "QDRANT_URL": {
        "expected_host_fragment": "qdrant.qdrant.svc",
        "expected_port": 6333,
    },
    "EMBEDDING_BASE_URL": {
        "expected_host_fragment": "embedding-service",
        "expected_port": 8002,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastAPI Kubernetes 챗봇 런타임 env 설정을 점검합니다."
    )
    parser.add_argument(
        "--env-file",
        default=K8S_ENV_EXAMPLE_PATH,
        help="점검할 Kubernetes runtime env 파일 경로",
    )
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="예시 파일 점검 시 Secret placeholder 값을 허용합니다.",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    return parser


def load_env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        values[key.strip()] = value.strip()
    return values


def check_k8s_runtime_env(
    values: dict[str, str],
    allow_placeholders: bool = False,
) -> dict:
    checks = []
    checks.extend(_check_exact_values(values))
    checks.extend(_check_secret_values(values, allow_placeholders))
    checks.extend(_check_config_values(values, allow_placeholders))
    checks.extend(_check_openai_model_allowlist(values, allow_placeholders))
    checks.extend(_check_k8s_service_urls(values))
    checks.extend(_check_disabled_spring_evidence_lookup(values))

    failures = [check for check in checks if check["status"] == "FAIL"]
    return {
        "checkStatus": "PASS" if not failures else "FAIL",
        "failureCount": len(failures),
        "checks": checks,
    }


def _check_exact_values(values: dict[str, str]) -> list[dict]:
    checks: list[dict] = []
    for key, expected_value in REQUIRED_EXACT_VALUES.items():
        actual_value = values.get(key)
        checks.append(
            _build_check(
                name=key,
                status="PASS" if actual_value == expected_value else "FAIL",
                reason=(
                    None
                    if actual_value == expected_value
                    else f"{key} 값은 {expected_value}이어야 합니다."
                ),
                expected=expected_value,
                actual=actual_value,
            )
        )
    return checks


def _check_secret_values(
    values: dict[str, str],
    allow_placeholders: bool,
) -> list[dict]:
    checks: list[dict] = []
    for key in sorted(REQUIRED_SECRET_KEYS):
        value = values.get(key, "")
        is_placeholder = value.startswith(PLACEHOLDER_PREFIX)
        is_valid = bool(value) and (allow_placeholders or not is_placeholder)
        checks.append(
            _build_check(
                name=key,
                status="PASS" if is_valid else "FAIL",
                reason=_build_secret_failure_reason(
                    key,
                    value=value,
                    allow_placeholders=allow_placeholders,
                    is_placeholder=is_placeholder,
                ),
                expected="non-empty secret reference",
                actual=_mask_secret_value(value),
            )
        )
    return checks


def _check_config_values(
    values: dict[str, str],
    allow_placeholders: bool,
) -> list[dict]:
    checks: list[dict] = []
    for key in sorted(REQUIRED_CONFIG_KEYS):
        value = values.get(key, "")
        is_placeholder = value.startswith(CONFIG_PLACEHOLDER_PREFIX)
        is_valid = bool(value) and (allow_placeholders or not is_placeholder)
        checks.append(
            _build_check(
                name=key,
                status="PASS" if is_valid else "FAIL",
                reason=_build_config_failure_reason(
                    key,
                    value=value,
                    allow_placeholders=allow_placeholders,
                    is_placeholder=is_placeholder,
                ),
                expected="non-empty runtime value",
                actual=_mask_config_value(value),
            )
        )
    return checks


def _check_openai_model_allowlist(
    values: dict[str, str],
    allow_placeholders: bool,
) -> list[dict]:
    model = values.get("LLM_MODEL", "")
    allowed_models_value = values.get("LLM_ALLOWED_MODELS", "")
    model_is_placeholder = model.startswith(CONFIG_PLACEHOLDER_PREFIX)
    allowed_is_placeholder = allowed_models_value.startswith(CONFIG_PLACEHOLDER_PREFIX)
    should_skip_placeholder = (
        allow_placeholders
        and model_is_placeholder
        and allowed_is_placeholder
    )
    allowed_models = _parse_config_list(allowed_models_value)
    is_valid = (
        should_skip_placeholder
        or (
            bool(model)
            and not model_is_placeholder
            and bool(allowed_models)
            and model in allowed_models
        )
    )
    return [
        _build_check(
            name="LLM_MODEL_ALLOWLIST",
            status="PASS" if is_valid else "FAIL",
            reason=(
                None
                if is_valid
                else "LLM_MODEL 값은 LLM_ALLOWED_MODELS에 포함되어야 합니다."
            ),
            expected="LLM_MODEL included in LLM_ALLOWED_MODELS",
            actual=_mask_config_value(allowed_models_value),
        )
    ]


def _parse_config_list(value: str) -> set[str]:
    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


def _build_secret_failure_reason(
    key: str,
    value: str,
    allow_placeholders: bool,
    is_placeholder: bool,
) -> str | None:
    if value and (allow_placeholders or not is_placeholder):
        return None
    if not value:
        return f"{key} 값은 Kubernetes Secret에서 주입되어야 합니다."
    return f"{key} placeholder는 실제 배포 env에서 사용할 수 없습니다."


def _build_config_failure_reason(
    key: str,
    value: str,
    allow_placeholders: bool,
    is_placeholder: bool,
) -> str | None:
    if value and (allow_placeholders or not is_placeholder):
        return None
    if not value:
        return f"{key} 값은 실제 런타임 설정으로 주입되어야 합니다."
    return f"{key} placeholder는 실제 배포 env에서 사용할 수 없습니다."


def _check_k8s_service_urls(values: dict[str, str]) -> list[dict]:
    checks: list[dict] = []
    for key, rule in K8S_SERVICE_URL_KEYS.items():
        value = values.get(key, "")
        parsed = urlparse(value)
        host = parsed.hostname or ""
        port = parsed.port
        is_valid = (
            parsed.scheme in {"http", "https"}
            and host not in LOCAL_HOSTS
            and rule["expected_host_fragment"] in host
            and port == rule["expected_port"]
        )
        checks.append(
            _build_check(
                name=key,
                status="PASS" if is_valid else "FAIL",
                reason=(
                    None
                    if is_valid
                    else (
                        f"{key} 값은 Kubernetes Service DNS와 "
                        f"{rule['expected_port']} 포트를 사용해야 합니다."
                    )
                ),
                expected=f"*{rule['expected_host_fragment']}*:{rule['expected_port']}",
                actual=value,
            )
        )
    return checks


def _check_disabled_spring_evidence_lookup(values: dict[str, str]) -> list[dict]:
    value = values.get("EVIDENCE_LOOKUP_ENABLED")
    is_valid = value == "false"
    return [
        _build_check(
            name="EVIDENCE_LOOKUP_ENABLED",
            status="PASS" if is_valid else "FAIL",
            reason=(
                None
                if is_valid
                else "현재 구조는 FastAPI가 read-only RDB View를 직접 조회합니다."
            ),
            expected="false",
            actual=value,
        )
    ]


def _build_check(
    name: str,
    status: str,
    reason: str | None,
    expected: object,
    actual: object,
) -> dict:
    check = {
        "name": name,
        "status": status,
        "expected": expected,
        "actual": actual,
    }
    if reason:
        check["reason"] = reason
    return check


def _mask_secret_value(value: str) -> str:
    if not value:
        return ""
    if value.startswith(PLACEHOLDER_PREFIX):
        return value
    return "<set>"


def _mask_config_value(value: str) -> str:
    if not value:
        return ""
    if value.startswith(CONFIG_PLACEHOLDER_PREFIX):
        return value
    return "<set>"


def format_text_result(result: dict) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"failureCount={result['failureCount']}",
    ]
    for check in result["checks"]:
        line = (
            f"{check['name']}: "
            f"status={check['status']} "
            f"expected={check['expected']} "
            f"actual={check['actual']}"
        )
        if "reason" in check:
            line = f"{line} reason={check['reason']}"
        lines.append(line)
    return "\n".join(lines)


def format_json_result(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


def main(
    argv: list[str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    args = build_parser().parse_args(argv)

    try:
        values = load_env_values(Path(args.env_file))
        result = check_k8s_runtime_env(
            values,
            allow_placeholders=args.allow_placeholders,
        )
    except Exception as exc:
        print(f"Kubernetes runtime env 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0 if result["checkStatus"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
