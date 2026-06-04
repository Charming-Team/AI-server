from pydantic import BaseModel, ConfigDict, Field


class DelayPredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    order_id: int = Field(alias="orderId")
    predicted_delay_hours: float = Field(alias="predictedDelayHours")
    model_name: str = Field(alias="modelName")
