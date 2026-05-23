from app.features.chat.source_url_policy import normalize_internal_url


def has_invalid_navigation_url(url: object) -> bool:
    if not isinstance(url, str):
        return False
    if not url.strip():
        return False
    return normalize_internal_url(url) is None


def has_navigation_target(
    url: object,
    reference_type: object,
    reference_id: object,
) -> bool:
    if isinstance(url, str) and normalize_internal_url(url) is not None:
        return True

    return has_reference_navigation_target(reference_type, reference_id)


def has_reference_navigation_target(
    reference_type: object,
    reference_id: object,
) -> bool:
    return (
        isinstance(reference_type, str)
        and bool(reference_type.strip())
        and _has_positive_reference_id(reference_id)
    )


def _has_positive_reference_id(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, str):
        stripped_value = value.strip()
        return stripped_value.isdigit() and int(stripped_value) > 0
    return False
