import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta


@dataclass(frozen=True)
class TargetFilter:
    target_type: str
    target_code: str


@dataclass(frozen=True)
class TargetPattern:
    target_type: str
    pattern: re.Pattern[str]


class QueryFilterExtractor:
    _default_limit = 5
    _count_query_limit = 50
    _line_overview_query_limit = 50
    _explicit_date_pattern = re.compile(
        r"(?<!\d)(20\d{2})[-./](0?[1-9]|1[0-2])[-./](0?[1-9]|[12]\d|3[01])(?!\d)"
    )
    _target_patterns = (
        TargetPattern(
            target_type="ORDER",
            pattern=re.compile(r"(?<![A-Z0-9])ORD-\d{6}-\d+(?![A-Z0-9])", re.IGNORECASE),
        ),
        TargetPattern(
            target_type="LINE",
            pattern=re.compile(
                r"(?<![A-Z0-9])LINE-[A-Z0-9]+(?:-[A-Z0-9]+)*(?![A-Z0-9-])",
                re.IGNORECASE,
            ),
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

    def extract_filters(
        self,
        question: str,
        reference_datetime: datetime | None = None,
    ) -> dict:
        target = self.extract_target(question)
        date_range = self.extract_date_range(question, reference_datetime)
        return {
            "limit": self._extract_limit(question),
            "fromDate": date_range[0],
            "toDate": date_range[1],
            "targetType": target.target_type if target else None,
            "targetCode": target.target_code if target else None,
        }

    def _extract_limit(self, question: str) -> int:
        if self._is_count_question(question):
            return self._count_query_limit
        if self._is_line_overview_question(question):
            return self._line_overview_query_limit
        return self._default_limit

    def _is_count_question(self, question: str) -> bool:
        compact_question = self._compact(question.casefold())
        return any(
            term in compact_question
            for term in (
                "몇개",
                "몇개의",
                "몇대",
                "개수",
                "총몇",
                "총수",
                "전체몇",
            )
        )

    def _is_line_overview_question(self, question: str) -> bool:
        compact_question = self._compact(question.casefold())
        has_line_term = any(term in compact_question for term in ("라인", "공정", "line"))
        if not has_line_term:
            return False

        running_terms = (
            "가동중",
            "가동중인",
            "현재가동",
            "운영중",
            "운영중인",
            "running",
        )
        if any(term in compact_question for term in running_terms):
            return True

        strong_overview_terms = (
            "구성",
            "목록",
            "리스트",
        )
        if any(term in compact_question for term in strong_overview_terms):
            return True

        broad_overview_terms = (
            "전체라인",
            "라인전체",
            "생산라인전체",
            "공정라인전체",
        )
        if (
            any(term in compact_question for term in broad_overview_terms)
            and not self._has_bottleneck_term(compact_question)
        ):
            return True

        status_terms = ("상태", "현황")
        return (
            any(term in compact_question for term in status_terms)
            and not self._has_bottleneck_term(compact_question)
        )

    def _has_bottleneck_term(self, compact_question: str) -> bool:
        return any(
            term in compact_question
            for term in (
                "병목",
                "대기",
                "지연",
                "위험",
                "문제",
                "원인",
                "이상",
            )
        )

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

    def extract_date_range(
        self,
        question: str,
        reference_datetime: datetime | None = None,
    ) -> tuple[str | None, str | None]:
        explicit_dates = self._extract_explicit_dates(question)
        if len(explicit_dates) >= 2:
            start_date, end_date = sorted(explicit_dates[:2])
            return start_date.isoformat(), end_date.isoformat()
        if len(explicit_dates) == 1:
            target_date = explicit_dates[0]
            return target_date.isoformat(), target_date.isoformat()

        reference_date = self._reference_date(reference_datetime)
        compact_question = question.replace(" ", "")
        if "오늘" in compact_question:
            return reference_date.isoformat(), reference_date.isoformat()
        if "내일" in compact_question:
            target_date = reference_date + timedelta(days=1)
            return target_date.isoformat(), target_date.isoformat()
        if "어제" in compact_question:
            target_date = reference_date - timedelta(days=1)
            return target_date.isoformat(), target_date.isoformat()
        if "이번주" in compact_question:
            return self._week_range(reference_date, week_offset=0)
        if "다음주" in compact_question:
            return self._week_range(reference_date, week_offset=1)
        if "이번달" in compact_question or "이번월" in compact_question:
            return self._month_range(reference_date, month_offset=0)
        if "다음달" in compact_question or "다음월" in compact_question:
            return self._month_range(reference_date, month_offset=1)
        return None, None

    def _extract_explicit_dates(self, question: str) -> list[date]:
        dates: list[date] = []
        for match in self._explicit_date_pattern.finditer(question):
            try:
                dates.append(
                    date(
                        int(match.group(1)),
                        int(match.group(2)),
                        int(match.group(3)),
                    )
                )
            except ValueError:
                continue
        return dates

    def _reference_date(self, reference_datetime: datetime | None) -> date:
        if reference_datetime is None:
            return date.today()
        return reference_datetime.date()

    def _week_range(self, reference_date: date, week_offset: int) -> tuple[str, str]:
        start_date = reference_date - timedelta(days=reference_date.weekday())
        start_date += timedelta(days=week_offset * 7)
        end_date = start_date + timedelta(days=6)
        return start_date.isoformat(), end_date.isoformat()

    def _month_range(self, reference_date: date, month_offset: int) -> tuple[str, str]:
        month_index = reference_date.month - 1 + month_offset
        year = reference_date.year + month_index // 12
        month = month_index % 12 + 1
        start_date = date(year, month, 1)
        if month == 12:
            next_month_start = date(year + 1, 1, 1)
        else:
            next_month_start = date(year, month + 1, 1)
        end_date = next_month_start - timedelta(days=1)
        return start_date.isoformat(), end_date.isoformat()

    def _compact(self, text: str) -> str:
        return "".join(text.split())
