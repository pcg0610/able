from pydantic import BaseModel
from block.enums import BlockType

class BlockDto(BaseModel):
    id: str
    type: BlockType
    position: str
    args: dict
        
class EdgeDto(BaseModel):
    id: str
    source: str
    target: str

class TrainRequestDto(BaseModel):
    project_id: str
    epoch: int
    batch_size: int
    blocks: list[BlockDto]
    edges: list[EdgeDto]