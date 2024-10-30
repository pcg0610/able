from pydantic import BaseModel
from block.enums import BlockType

class BlockDto(BaseModel):
    block_id: str
    type: BlockType
    position: str
    args: dict
        
class EdgeDto(BaseModel):
    edge_id: str
    source: str
    target: str

class TrainRequestDto(BaseModel):
    project_name: str
    epoch: int
    batch_size: int
    blocks: list[BlockDto]
    edges: list[EdgeDto]
    
class TrainResponseDto(BaseModel):
    pass