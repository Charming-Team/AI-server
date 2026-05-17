import re
from dataclasses import dataclass
from re import Pattern


@dataclass(frozen=True)
class SensitivePatternRule:
    name: str
    pattern: Pattern[str]


class SensitivePatternPolicy:
    _long_base64_candidate_pattern = re.compile(
        r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{80,}={0,2}(?![A-Za-z0-9+/=])"
    )
    _rules: tuple[SensitivePatternRule, ...] = (
        SensitivePatternRule(
            name="jwt",
            pattern=re.compile(
                r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
            ),
        ),
        SensitivePatternRule(
            name="bearer_token",
            pattern=re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b", re.IGNORECASE),
        ),
        SensitivePatternRule(
            name="private_key",
            pattern=re.compile(
                r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----",
                re.IGNORECASE,
            ),
        ),
        SensitivePatternRule(
            name="aws_access_key",
            pattern=re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        ),
        SensitivePatternRule(
            name="openai_style_api_key",
            pattern=re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        ),
        SensitivePatternRule(
            name="github_token",
            pattern=re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b"),
        ),
        SensitivePatternRule(
            name="secret_assignment",
            pattern=re.compile(
                r"\b(?:api[_-]?key|secret|token|password|passwd|pwd)"
                r"\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{16,}['\"]?",
                re.IGNORECASE,
            ),
        ),
    )

    def contains_sensitive_pattern(self, text: str) -> bool:
        return any(
            rule.pattern.search(text) is not None for rule in self._rules
        ) or self._contains_long_base64_secret(text)

    def _contains_long_base64_secret(self, text: str) -> bool:
        return any(
            self._is_suspicious_base64_candidate(candidate)
            for candidate in self._long_base64_candidate_pattern.findall(text)
        )

    def _is_suspicious_base64_candidate(self, candidate: str) -> bool:
        unpadded_candidate = candidate.rstrip("=")
        return (
            any(char.islower() for char in unpadded_candidate)
            and any(char.isupper() for char in unpadded_candidate)
            and any(char.isdigit() for char in unpadded_candidate)
            and len(set(unpadded_candidate)) >= 10
        )
