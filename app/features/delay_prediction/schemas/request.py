from pydantic import BaseModel, ConfigDict, Field


class DelayPredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    order_id: int = Field(alias="orderId", ge=1)
