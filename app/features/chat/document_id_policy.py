import re

MAX_DOCUMENT_ID_LENGTH = 200

_DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def document_id_format_error(document_id: str) -> str | None:
    if len(document_id) > MAX_DOCUMENT_ID_LENGTH:
        return f"문서 ID는 최대 {MAX_DOCUMENT_ID_LENGTH}자까지 허용됩니다."

    if _DOCUMENT_ID_PATTERN.fullmatch(document_id):
        return None

    return "문서 ID는 영문, 숫자, '.', '_', ':', '-'만 사용할 수 있습니다."


def is_safe_document_id(document_id: str) -> bool:
    return document_id_format_error(document_id) is None
