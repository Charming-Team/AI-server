import pytest

from app.features.chat.sensitive_pattern_policy import SensitivePatternPolicy


@pytest.mark.parametrize(
    "text",
    [
        (
            "Authorization: Bearer "
            "abcDEF1234567890abcDEF1234567890abcDEF1234567890"
        ),
        (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ),
        "-----BEGIN PRIVATE KEY-----",
        "AKIAIOSFODNN7EXAMPLE",
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "ghp_abcdefghijklmnopqrstuvwxyz123456",
        "api_key=abcdefghijklmnopqrstuvwxyz123456",
        (
            "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXphYmNkZWZnaGlqa2xtbm9wcXJzdHV2"
            "d3h5ejEyMzQ1Njc4OTBhYmNkZWZnaGlqa2xtbm9w"
        ),
    ],
)
def test_sensitive_pattern_policy_blocks_secret_like_patterns(text: str) -> None:
    policy = SensitivePatternPolicy()

    assert policy.contains_sensitive_pattern(text) is True


def test_sensitive_pattern_policy_allows_business_text() -> None:
    policy = SensitivePatternPolicy()

    assert policy.contains_sensitive_pattern("LINE-A01 자재 부족 리스크를 확인합니다.") is False


def test_sensitive_pattern_policy_allows_repeated_long_text() -> None:
    policy = SensitivePatternPolicy()

    assert policy.contains_sensitive_pattern("A" * 200) is False
