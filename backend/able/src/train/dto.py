from pydantic import BaseModel
from src.block.schemas import Block, Edge

class TrainRequestDto(BaseModel):
    project_name: str
    epoch: int
    batch_size: int
    data: Block
    interpreter: Block
    loss: Block
    optimizer: Block
    transforms: tuple[Block]
    blocks: tuple[Block]
    edges: tuple[Edge]

    class Config:
        frozen = True

class TrainResponseDto(BaseModel):
    pass