import pytest

from app.features.chat.navigation_target_policy import (
    has_invalid_navigation_url,
    has_navigation_target,
    has_reference_navigation_target,
)


@pytest.mark.parametrize(
    "url",
    [
        "/reports/20",
        " /reports/20?mode=read ",
    ],
)
def test_has_navigation_target_accepts_internal_url(url: str) -> None:
    assert has_navigation_target(url, None, None) is True
    assert has_invalid_navigation_url(url) is False


@pytest.mark.parametrize(
    "url",
    [
        "https://external.example/reports/20",
        "//external.example/reports/20",
    ],
)
def test_has_invalid_navigation_url_blocks_external_url(url: str) -> None:
    assert has_invalid_navigation_url(url) is True


@pytest.mark.parametrize(
    ("reference_id", "expected"),
    [
        (20, True),
        ("20", True),
        (0, False),
        (-1, False),
        ("0", False),
        (True, False),
        (None, False),
    ],
)
def test_has_reference_navigation_target_requires_positive_reference_id(
    reference_id: object,
    expected: bool,
) -> None:
    assert has_reference_navigation_target("REPORT", reference_id) is expected


def test_has_navigation_target_accepts_reference_metadata_without_url() -> None:
    assert has_navigation_target(None, "REPORT", 20) is True


def test_has_navigation_target_rejects_missing_target() -> None:
    assert has_navigation_target(None, None, None) is False
