from datetime import datetime
from uuid import uuid4, UUID
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field, conint
from typing import Any, Optional

class ImmutableBaseModel(BaseModel):
    class Config:
        frozen = True

class Response(BaseModel):
    status_code: conint(gt=99, lt=600)
    time_stamp: datetime  = Field(default_factory=lambda: datetime.now(ZoneInfo("Asia/Seoul")))
    tracking_id: UUID  = Field(default_factory=uuid4)
    data: Optional[Any] = None