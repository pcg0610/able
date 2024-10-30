from typing import Dict
from src.block.enums import BlockType
from pydantic import BaseModel


class Block(BaseModel):
    block_id: str
    type: BlockType
    position: str
    args: Dict[str, str]

    class Config:
        frozen = True

class Edge(BaseModel):
    edge_id: str
    source: str
    target: str

    class Config:
        frozen = True

class SearchBlockResponse(BaseModel):
    block: Block