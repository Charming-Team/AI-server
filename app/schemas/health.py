from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str


class ReadinessComponent(BaseModel):
    name: str
    enabled: bool
    configured: bool
    code: str | None = None
    reason: str | None = None


class ReadinessResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    components: list[ReadinessComponent]
