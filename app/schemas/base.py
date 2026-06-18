from pydantic import BaseModel, ConfigDict


def to_camel(value: str) -> str:
    """Convert internal snake_case field names to external camelCase names."""
    head, *tail = value.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


class ApiSchema(BaseModel):
    """Base schema for S-MAP FastAPI models.

    Python code uses snake_case attributes, while external API JSON is serialized
    with camelCase aliases.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )
