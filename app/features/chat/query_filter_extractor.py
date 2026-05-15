import re


class QueryFilterExtractor:
    _target_code_patterns = (
        re.compile(r"(?<![A-Z0-9])ORD-\d{6}-\d+(?![A-Z0-9])", re.IGNORECASE),
        re.compile(r"(?<![A-Z0-9])LINE-[A-Z]\d{2}(?![A-Z0-9])", re.IGNORECASE),
        re.compile(r"(?<![A-Z0-9])PROD-[A-Z0-9-]+(?![A-Z0-9])", re.IGNORECASE),
        re.compile(r"(?<![A-Z0-9])MAT-[A-Z0-9-]+(?![A-Z0-9])", re.IGNORECASE),
        re.compile(r"(?<![A-Z0-9])RM-[A-Z0-9-]+(?![A-Z0-9])", re.IGNORECASE),
    )

    def extract_filters(self, question: str) -> dict:
        return {
            "limit": 5,
            "fromDate": None,
            "toDate": None,
            "targetCode": self.extract_target_code(question),
        }

    def extract_target_code(self, question: str) -> str | None:
        for pattern in self._target_code_patterns:
            match = pattern.search(question)
            if match:
                return match.group(0).upper()
        return None
