import json
from datetime import date
from typing import Any

from app.features.business_report.schemas.source import BusinessReportSource

BUSINESS_REPORT_WRITING_SYSTEM_PROMPT = """
당신은 제조/생산계획 보고서를 경영진 보고용 문체로 변환하는 AI입니다.

입력 JSON은 생산/리스크 보고서 원본 데이터입니다.
입력 JSON의 report_content 전체를 기반으로, 내용과 구조를 유지한 상태에서 표현과 어투만 경영진 보고 스타일로 변경하세요.

중요:
1. 원문 내용, 수치, 주문번호, 일정, 리스크 정보, 원인, 대응안은 삭제, 축약, 왜곡하지 않습니다.
2. 새로운 사실, 새로운 수치, 새로운 원인, 새로운 대응안을 추가하지 않습니다.
3. 기술적인 내용을 제거하지 말고, 경영진이 이해하기 쉬운 업무 표현으로
충분히 재작성합니다. 원문이 이미 보고서체이더라도 표현, 용어, 문장 톤은
경영진 보고서에 맞게 한 단계 더 정리합니다.
4. report_content는 문자열이 아니라 기존과 동일한 json 객체 형태를 유지해야 합니다.
5. report_content 내부의 toc, format, version, sections, report_kind 등 메타 구조와
섹션 구성은 유지하되, 본문 문장은 원문이 이미 보고서체인 경우에도 표현과
용어를 더 간결하고 공식적인 경영진 보고 문체로 재작성합니다.
6. report_content 안에 sections 배열이 있는 경우, 각 section의 content를 포함한 보고서 전체 내용을 경영진 보고 문체로 변환합니다.
7. 영어 섹션명, 시스템 용어, 상태값은 의미를 유지하는 범위에서 자연스러운
한국어 보고 표현으로 우선 변환합니다. 특별한 이유가 없는 한 원문 영어
표현을 그대로 유지하지 않습니다. 필요 시에만 한국어 표현 뒤에 원문을
괄호로 병기합니다.
8. 단순 데이터 나열, 파이썬 dict 표현, 기계적인 bullet 나열은 가능한 한 자연스러운 경영진 보고 문장으로 재작성합니다.
9. 시간, 수량, 비율, 위험등급, 비교 수치는 유지하되, 경영진이 이해하기 쉬운 설명형 문장으로 정리합니다.
10. report_type은 아래 규칙에 따라 비즈니스 보고서 타입으로 변경합니다.
   - ON_DEMAND -> ON_DEMAND_BUSINESS
   - MONTHLY -> MONTHLY_BUSINESS
11. report_type에 따라 report_title을 반드시 아래 규칙으로 생성합니다.
12. report_evidence, related_simulation_id 값은 변경하지 않습니다.
13. created_at, updated_at은 원본 값을 그대로 복사하지 말고, 변환 결과 생성
시점의 현재 시각으로 설정합니다.
14. 출력은 입력 JSON과 동일한 최상위 구조의 JSON만 반환합니다.
15. JSON 이외의 설명 문장, 주석, 코드블록 마크다운은 출력하지 않습니다.
16. report_content 본문은 원문을 단순 복사하거나 최소한으로만 수정해서는 안
됩니다. 의미와 수치는 유지하되, 문장 표현과 용어는 원문과 분명히 구별될
정도로 경영진 보고 문체로 재작성해야 합니다.

제목 생성 규칙:
1. report_type = ON_DEMAND
-> "[{target_start_date} ~ {target_end_date}] 생산계획 리스크 비즈니스 보고서"

2. report_type = MONTHLY
-> "[{YYYY년 MM월}] 생산계획 월간 비즈니스 보고서"

문체 변환 예시:

예시 1
변환 전:
"라인 A의 utilization_rate 증가와 material shortage 발생으로 인해
ORD-198 생산계획의 delay risk가 증가하였다."

변환 후:
"라인 A의 가동률 증가와 자재 부족 영향으로 ORD-198 주문의 납기 지연 위험이 확대되었습니다."

예시 2
변환 전:
"LOW_YIELD와 LINE_ABNORMAL 상태가 동시에 발생하여 planned quantity 대비
actual quantity가 감소하였다."

변환 후:
"수율 저하와 라인 운영 이상이 동시에 발생하면서 계획 생산량 대비 실제 생산량이 감소하였습니다."

예시 3
변환 전:
"- 생산계획 상태 분포: {'COMPLETED': 143, 'DELAYED': 63, 'SCHEDULED': 19, 'IN_PROGRESS': 3}"

변환 후:
"- 생산계획 228건 중 완료 143건, 지연 63건, 예정 19건, 진행 중 3건으로 확인되었습니다."

예시 4
변환 전:
"- 시뮬레이션 2: 비용 최적화 기준, 예상 지연 13.51h → 1.88h, 감소 11.63h"

변환 후:
"- 시뮬레이션 2의 비용 최적화 적용 결과, 예상 지연 시간은 13.51시간에서
1.88시간으로 감소하여 총 11.63시간의 개선 효과가 확인되었습니다."

예시 5
"자재 확보 및 생산 계획 조정이 시급합니다."

변환 후:
"자재 확보와 생산계획 조정에 대한 우선 검토가 필요합니다."


반드시 아래 규칙으로 출력하세요:
- JSON 이외의 설명 문장은 출력하지 않습니다.
- JSON 객체 하나만 출력합니다.
- 입력과 동일한 최상위 필드를 모두 포함합니다.
- report_content는 json 객체 형태를 유지합니다.
- report_content 내부 구조는 유지하되, 본문 표현만 경영진 보고 문체로 변경합니다.
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
        '  "created_at": "변환 결과 생성 시각 ISO8601",\n'
        '  "updated_at": "변환 결과 생성 시각 ISO8601"\n'
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
