from typing import Dict, List
from src.block.enums import BlockType
from src.response.schemas import ImmutableBaseModel

class Arg(ImmutableBaseModel):
    name: str
    is_required: bool

class Block(ImmutableBaseModel):
    name: str
    type: BlockType
    position: str
    args: List[Arg]

class Edge(ImmutableBaseModel):
    edge_id: str
    source: str
    target: str

class BlockResponse(ImmutableBaseModel):
    block: Block

class BlocksResponse(ImmutableBaseModel):
    blocks: List[Block]