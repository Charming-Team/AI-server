import argparse
import json
import sys
from typing import TextIO

from app.api.v1.routes.health import build_readiness_components
from app.core.config import Settings
from app.features.chat.runtime_mode import build_chat_runtime_mode
from app.features.chat.schemas import ChatErrorCode

REQUIRED_COMPONENT_OPTIONS = {
    "rdbEvidence": {
        "code": ChatErrorCode.CHAT_EVIDENCE_004,
        "reason": "RDB Evidence View 사용이 요구되지만 활성화되어 있지 않습니다.",
    },
    "qdrantSearch": {
        "code": ChatErrorCode.CHAT_QDRANT_001,
        "reason": "Qdrant 검색 사용이 요구되지만 활성화되어 있지 않습니다.",
    },
    "ragSearchPipeline": {
        "code": ChatErrorCode.CHAT_EMBEDDING_001,
        "reason": "Vector 검색 사용이 요구되지만 검색 파이프라인이 준비되지 않았습니다.",
    },
    "llm": {
        "code": ChatErrorCode.CHAT_LLM_001,
        "reason": "LLM 답변 생성이 요구되지만 활성화되어 있지 않습니다.",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastAPI 챗봇 런타임 설정의 readiness 상태를 점검합니다."
    )
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. 생략하면 기본 .env 설정을 사용합니다.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON",
    )
    parser.add_argument(
        "--require-rdb-evidence",
        action="store_true",
        help="RDB Evidence View 파이프라인이 활성화되어 있어야 합니다.",
    )
    parser.add_argument(
        "--require-vector-search",
        action="store_true",
        help="Qdrant 검색과 RAG 검색 파이프라인이 활성화되어 있어야 합니다.",
    )
    parser.add_argument(
        "--require-llm-generation",
        action="store_true",
        help="LLM 답변 생성이 활성화되어 있어야 합니다.",
    )
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    if args.env_file:
        return Settings(_env_file=args.env_file)
    return Settings()


def build_readiness_result(
    settings: Settings,
    required_components: list[str] | None = None,
) -> dict:
    components = build_readiness_components(settings)
    component_items = [
        component.model_dump(mode="json", exclude_none=True)
        for component in components
    ]
    requirement_failures = build_requirement_failures(
        component_items,
        required_components or [],
    )
    is_ready = (
        all(component.configured for component in components)
        and not requirement_failures
    )
    return {
        "status": "ready" if is_ready else "not_ready",
        "runtimeMode": build_chat_runtime_mode(settings).model_dump(
            mode="json",
            by_alias=True,
        ),
        "components": component_items,
        "requirementFailures": requirement_failures,
    }


def build_required_components(args: argparse.Namespace) -> list[str]:
    required_components: list[str] = []
    if args.require_rdb_evidence:
        required_components.append("rdbEvidence")
    if args.require_vector_search:
        required_components.extend(["qdrantSearch", "ragSearchPipeline"])
    if args.require_llm_generation:
        required_components.append("llm")
    return required_components


def build_requirement_failures(
    components: list[dict],
    required_components: list[str],
) -> list[dict]:
    component_by_name = {component["name"]: component for component in components}
    failures: list[dict] = []
    for component_name in required_components:
        component = component_by_name.get(component_name)
        if component is None:
            failures.append(_build_requirement_failure(component_name))
            continue
        if component.get("enabled") is True and component.get("configured") is True:
            continue
        failures.append(
            _build_requirement_failure(
                component_name,
                component.get("code"),
                component.get("reason"),
            )
        )
    return failures


def _build_requirement_failure(
    component_name: str,
    code: str | None = None,
    reason: str | None = None,
) -> dict:
    option = REQUIRED_COMPONENT_OPTIONS[component_name]
    return {
        "name": component_name,
        "code": code or option["code"],
        "reason": reason or option["reason"],
    }


def format_text_result(result: dict) -> str:
    runtime_mode = result["runtimeMode"]
    lines = [
        f"status={result['status']}",
        f"apiPrefix={runtime_mode['apiPrefix']}",
        f"groundingMode={runtime_mode['groundingMode']}",
        f"answerMode={runtime_mode['answerMode']}",
        f"ragSearchMode={runtime_mode['ragSearchMode']}",
        "enabledGroundingSources="
        f"{','.join(runtime_mode['enabledGroundingSources'])}",
        (
            "expectedLlmSkippedReason="
            f"{runtime_mode['expectedLlmSkippedReason']}"
        ),
    ]
    for component in result["components"]:
        line = (
            f"{component['name']}: "
            f"enabled={component['enabled']} "
            f"configured={component['configured']}"
        )
        if "code" in component:
            line = f"{line} code={component['code']}"
        if "reason" in component:
            line = f"{line} reason={component['reason']}"
        lines.append(line)
    for failure in result["requirementFailures"]:
        lines.append(
            "requirementFailure: "
            f"name={failure['name']} "
            f"code={failure['code']} "
            f"reason={failure['reason']}"
        )
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
        settings = build_settings(args)
        result = build_readiness_result(
            settings,
            required_components=build_required_components(args),
        )
    except Exception as exc:
        print(f"챗봇 readiness 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
