from app.features.chat.schemas import ChatErrorCode


class ChatExternalServiceError(Exception):
    def __init__(
        self,
        status_code: int,
        code: ChatErrorCode,
        message: str,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)
