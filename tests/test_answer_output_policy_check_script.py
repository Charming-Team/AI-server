from io import StringIO

from app.features.chat.answer_output_policy import AnswerOutputPolicy
from app.features.chat.schemas import ChatErrorCode, SecurityStatus
from scripts import check_answer_output_policy


def test_check_answer_output_policy_passes_default_cases() -> None:
    result = check_answer_output_policy.check_answer_output_policy()

    assert result["checkStatus"] == "PASS"
    assert result["caseCount"] == 4
    assert result["passedCaseCount"] == 4
    assert result["failedCaseCount"] == 0
    assert [case["name"] for case in result["cases"]] == [
        "normalBusinessOutput",
        "promptInjectionOutput",
        "sensitiveOutput",
        "operatorFinancialOutput",
    ]


def test_check_answer_output_policy_reports_failed_case() -> None:
    cases = (
        check_answer_output_policy.OutputPolicyCase(
            name="wrongExpectation",
            answer="보고서 근거에 따르면 자재 부족이 주요 리스크입니다.",
            role="EXECUTIVE",
            expected_status=SecurityStatus.BLOCKED_SENSITIVE_REQUEST,
            expected_code=ChatErrorCode.CHAT_SECURITY_002,
        ),
    )

    result = check_answer_output_policy.check_answer_output_policy(
        AnswerOutputPolicy(),
        cases,
    )

    assert result["checkStatus"] == "FAIL"
    assert result["caseCount"] == 1
    assert result["passedCaseCount"] == 0
    assert result["failedCaseCount"] == 1
    assert result["cases"][0] == {
        "name": "wrongExpectation",
        "role": "EXECUTIVE",
        "expectedStatus": "BLOCKED_SENSITIVE_REQUEST",
        "actualStatus": None,
        "expectedCode": "CHAT_SECURITY_002",
        "actualCode": None,
        "passed": False,
    }


def test_check_answer_output_policy_formats_text_result() -> None:
    result = check_answer_output_policy.check_answer_output_policy()

    text = check_answer_output_policy.format_text_result(result)

    assert "status=PASS" in text
    assert "case=promptInjectionOutput passed=True" in text
    assert "actualStatus=BLOCKED_PROMPT_INJECTION" in text


def test_check_answer_output_policy_formats_json_result() -> None:
    result = check_answer_output_policy.check_answer_output_policy()

    text = check_answer_output_policy.format_json_result(result)

    assert '"checkStatus": "PASS"' in text
    assert '"actualCode": "CHAT_SECURITY_001"' in text


def test_check_answer_output_policy_main_returns_success() -> None:
    stdout = StringIO()

    exit_code = check_answer_output_policy.main(["--json"], stdout=stdout)

    assert exit_code == 0
    assert '"checkStatus": "PASS"' in stdout.getvalue()
