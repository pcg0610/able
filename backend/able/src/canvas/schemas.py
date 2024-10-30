from pydantic import BaseModel
from typing import Dict, Any

class BlockGraphResponse(BaseModel):
    data: Dict[str, Any]