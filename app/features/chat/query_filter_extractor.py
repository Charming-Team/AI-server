import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetFilter:
    target_type: str
    target_code: str


@dataclass(frozen=True)
class TargetPattern:
    target_type: str
    pattern: re.Pattern[str]


class QueryFilterExtractor:
    _target_patterns = (
        TargetPattern(
            target_type="ORDER",
            pattern=re.compile(r"(?<![A-Z0-9])ORD-\d{6}-\d+(?![A-Z0-9])", re.IGNORECASE),
        ),
        TargetPattern(
            target_type="LINE",
            pattern=re.compile(r"(?<![A-Z0-9])LINE-[A-Z]\d{2}(?![A-Z0-9])", re.IGNORECASE),
        ),
        TargetPattern(
            target_type="PRODUCT",
            pattern=re.compile(r"(?<![A-Z0-9])PROD-[A-Z0-9-]+(?![A-Z0-9])", re.IGNORECASE),
        ),
        TargetPattern(
            target_type="MATERIAL",
            pattern=re.compile(r"(?<![A-Z0-9])MAT-[A-Z0-9-]+(?![A-Z0-9])", re.IGNORECASE),
        ),
        TargetPattern(
            target_type="MATERIAL",
            pattern=re.compile(r"(?<![A-Z0-9])RM-[A-Z0-9-]+(?![A-Z0-9])", re.IGNORECASE),
        ),
    )

    def extract_filters(self, question: str) -> dict:
        target = self.extract_target(question)
        return {
            "limit": 5,
            "fromDate": None,
            "toDate": None,
            "targetType": target.target_type if target else None,
            "targetCode": target.target_code if target else None,
        }

    def extract_target_code(self, question: str) -> str | None:
        target = self.extract_target(question)
        if target is None:
            return None
        return target.target_code

    def extract_target_type(self, question: str) -> str | None:
        target = self.extract_target(question)
        if target is None:
            return None
        return target.target_type

    def extract_target(self, question: str) -> TargetFilter | None:
        for target_pattern in self._target_patterns:
            match = target_pattern.pattern.search(question)
            if match:
                return TargetFilter(
                    target_type=target_pattern.target_type,
                    target_code=match.group(0).upper(),
                )
        return None
