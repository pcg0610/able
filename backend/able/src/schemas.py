from pydantic import BaseModel
from typing import Any, Optional

class ResponseSchema(BaseModel):
    status_code: int
    detail: str
    data: Optional[Any] = None