from uuid import UUID
from pydantic import BaseModel, Field
from typing import List

# Pydantic's `Field` object allows you to specify metadata and validations for
# model atributes.
class Estimate(BaseModel):
    id: UUID = Field(
        title="The unique estimate ID.",
        description="A globally unique estimate id.",
        example="{12345678-1234-5678-1234-567812345678}",
    )
    value: float

    # Adding a "Config" class with a "schema_extra" parameter will render
    # "schema_extra" in the documenation anywhere the `Estimate` type is used.
    # It replaces the generic defaults used when showing examples, which allows
    # readers to see a more "real world" example.
    class Config:
        schema_extra = {"example": {"id": 100, "value": 42343.23}}


class EstimateRequest(BaseModel):
    times: int = Field(
        default=1,
        title="Estimate count",
        description="The number of estimates to generate.",
        ge=0,
        # You can also add an "estimate" parameter to the Field object to show
        # this default value when the field is rendered in the documentation.
        example="42",
    )


class EstimateResponse(BaseModel):
    estimates: List[Estimate]
