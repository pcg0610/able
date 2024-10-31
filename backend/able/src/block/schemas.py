from typing import Dict, List
from src.block.enums import BlockType
from src.response.schemas import ImmutableBaseModel


class Block(ImmutableBaseModel):
    name: str
    type: BlockType
    position: str
    args: Dict[str, str]

    class Config:
        frozen = True

class Edge(ImmutableBaseModel):
    edge_id: str
    source: str
    target: str

class BlockResponse(ImmutableBaseModel):
    block: Block

class BlocksResponse(ImmutableBaseModel):
    blocks: List[Block]