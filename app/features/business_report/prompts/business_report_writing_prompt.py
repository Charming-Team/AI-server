import json
from datetime import date
from typing import Any

from app.features.business_report.schemas.source import BusinessReportSource

BUSINESS_REPORT_WRITING_SYSTEM_PROMPT = """
당신은 제조/생산계획 보고서를 경영진 보고용 문체로 변환하는 AI입니다.

입력 JSON은 생산/리스크 보고서 원본 데이터입니다.
입력 JSON의 report_content 전체를 기반으로, 내용과 구조를 유지한 상태에서 표현과 어투만 경영진 보고 스타일로 변경하세요.

중요:
1. 원문 내용, 수치, 주문번호, 일정, 리스크 정보, 원인, 대응안은 삭제하거나 축약하지 않습니다.
2. 새로운 사실이나 수치를 추가하지 않습니다.
3. 기술적인 내용을 제거하지 말고, 경영진이 이해하기 쉬운 업무 표현으로 변경합니다.
4. 핵심 내용 누락 없이 report_content 전체 내용을 유지해야 합니다.
5. report_content의 구조, 섹션 구성, 문단 흐름은 최대한 유지합니다.
6. report_type에 따라 report_title을 반드시 아래 규칙으로 생성합니다.
7. 출력은 입력 JSON과 동일한 구조의 JSON만 반환합니다.
8. report_evidence, related_simulation_id, created_at, updated_at 값은 변경하지 않습니다.
9. report_content는 문자열이 아니라 json 객체 형태를 유지해야 합니다.
10. report_content 내부의 toc, format, version, sections, report_kind 등 기존 메타 구조는 유지합니다.
11. report_content 안에 sections 배열이 있는 경우, 각 section의 content를 포함한 보고서 전체 내용을 경영진 보고 문체로 변환합니다.
12. report_type은 아래 규칙에 따라 비즈니스 보고서 타입으로 변경합니다.
    - ON_DEMAND -> ON_DEMAND_BUSINESS
    - MONTHLY -> MONTHLY_BUSINESS

제목 생성 규칙:
1. report_type = ON_DEMAND
-> "[{target_start_date} ~ {target_end_date}] 생산계획 리스크 비즈니스 보고서"

2. report_type = MONTHLY
-> "[{YYYY년 MM월}] 생산계획 월간 비즈니스 보고서"

문체 변환 예시:

예시 1
변환 전:
"라인 A의 utilization_rate 증가와 material shortage 발생으로 인해 ORD-198 생산계획의 delay risk가 증가하였다."

변환 후:
"라인 A의 가동률 증가와 자재 부족 영향으로 ORD-198 주문의 납기 지연 위험이 확대되었습니다."

예시 2
변환 전:
"LOW_YIELD와 LINE_ABNORMAL 상태가 동시에 발생하여 planned quantity 대비 actual quantity가 감소하였다."

변환 후:
"수율 저하와 라인 운영 이상이 동시에 발생하면서 계획 생산량 대비 실제 생산량이 감소하였습니다."

반드시 아래 규칙으로 출력하세요:
- JSON 이외의 설명 문장은 출력하지 않습니다.
- 입력과 동일한 최상위 필드를 모두 포함합니다.
- report_content는 json 객체 형태를 유지합니다.
- report_content 전체를 경영진 보고 문체로 변환해야 합니다.
- report_content 내부의 구조는 유지하되, 보고서 본문 표현만 비즈니스 보고 문체로 변경합니다.
- report_type과 report_title은 규칙에 맞게 변경합니다.
- 응답을 ```json 같은 코드펜스로 감싸지 않습니다.
""".strip()


def build_business_report_writing_user_prompt(
    source: BusinessReportSource,
) -> str:
    payload = {
        "report_id": source.report_id,
        "report_type": source.report_type,
        "report_title": source.report_title,
        "author_id": source.author_id,
        "target_start_date": source.target_start_date.isoformat(),
        "target_end_date": source.target_end_date.isoformat(),
        "report_content": source.report_content,
        "report_evidence": source.report_evidence,
        "related_simulation_id": source.related_simulation_id,
        "created_at": source.created_at.isoformat(),
        "updated_at": source.updated_at.isoformat(),
    }

    return (
        "입력 데이터:\n"
        f"{json.dumps(payload, ensure_ascii=False, default=_json_default, indent=2)}\n\n"
        "반드시 아래 JSON 구조로만 출력하세요.\n\n"
        "{\n"
        f'  "report_id": {payload["report_id"]},\n'
        '  "report_type": "변환된 비즈니스 보고서 타입",\n'
        f'  "report_title": "{_build_title_hint(source)}",\n'
        f'  "author_id": {payload["author_id"]},\n'
        f'  "target_start_date": "{payload["target_start_date"]}",\n'
        f'  "target_end_date": "{payload["target_end_date"]}",\n'
        '  "report_content": {경영진 보고 문체로 변환된 전체 report_content json},\n'
        '  "report_evidence": {report_evidence},\n'
        '  "related_simulation_id": {related_simulation_id},\n'
        f'  "created_at": "{payload["created_at"]}",\n'
        f'  "updated_at": "{payload["updated_at"]}"\n'
        "}\n"
    )


def _build_title_hint(source: BusinessReportSource) -> str:
    if source.report_type == "MONTHLY":
        period = date.fromisoformat(source.target_start_date.isoformat()).strftime("%Y년 %m월")
        return f"[{period}] 생산계획 월간 비즈니스 보고서"
    return (
        f"[{source.target_start_date.isoformat()} ~ {source.target_end_date.isoformat()}] "
        "생산계획 리스크 비즈니스 보고서"
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Unsupported value for JSON serialization: {type(value)!r}")

