from pydantic import BaseModel
from src.block.enums import BlockType
from dataclasses import dataclass

@dataclass(frozen=True)
class BlockDto(BaseModel):
    block_id: str
    type: BlockType
    position: str
    args: dict

@dataclass(frozen=True)
class EdgeDto(BaseModel):
    edge_id: str
    source: str
    target: str

@dataclass(frozen=True)
class TrainRequestDto(BaseModel):
    project_name: str
    epoch: int
    batch_size: int
    data: BlockDto
    interpreter: BlockDto
    loss: BlockDto
    optimizer: BlockDto
    blocks: tuple[BlockDto]
    edges: tuple[EdgeDto]
    
class TrainResponseDto(BaseModel):
    pass