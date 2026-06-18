import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]
InternalTokenHeader = Annotated[str | None, Header(alias="X-Internal-Token")]


def verify_internal_api_token(
    settings: SettingsDep,
    x_internal_token: InternalTokenHeader = None,
) -> None:
    """Verify the internal Spring-to-FastAPI API token."""
    expected_token = settings.internal_api_token
    if not expected_token or not x_internal_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INTERNAL_AUTH_001",
                "message": "X-Internal-Token header is required.",
            },
        )

    if not hmac.compare_digest(x_internal_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INTERNAL_AUTH_002",
                "message": "X-Internal-Token header is invalid.",
            },
        )
