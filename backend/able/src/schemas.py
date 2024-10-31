from pydantic import BaseModel
from typing import Any, Optional

class ImmutableBaseModel(BaseModel):
    class Config:
        frozen = True

class ResponseSchema(ImmutableBaseModel):
    status_code: int
    detail: str
    data: Optional[Any] = None