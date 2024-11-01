from typing import Any
from src.block.enums import BlockType
from src.response.schemas import ImmutableBaseModel

class Arg(ImmutableBaseModel):
    name: str
    value: Any
    is_required: bool

class Block(ImmutableBaseModel):
    name: str
    type: BlockType
    args: list[Arg]

class Edge(ImmutableBaseModel):
    edge_id: str
    source: str
    target: str

class BlockResponse(ImmutableBaseModel):
    block: Block

class BlocksResponse(ImmutableBaseModel):
    blocks: list[Block]