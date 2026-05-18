import argparse
import json
import sys
from typing import TextIO

from app.api.v1.routes.health import build_readiness_components
from app.core.config import Settings


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
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    if args.env_file:
        return Settings(_env_file=args.env_file)
    return Settings()


def build_readiness_result(settings: Settings) -> dict:
    components = build_readiness_components(settings)
    is_ready = all(component.configured for component in components)
    return {
        "status": "ready" if is_ready else "not_ready",
        "components": [
            component.model_dump(mode="json", exclude_none=True)
            for component in components
        ],
    }


def format_text_result(result: dict) -> str:
    lines = [f"status={result['status']}"]
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
        result = build_readiness_result(settings)
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
