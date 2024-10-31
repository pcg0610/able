from typing import List
from src.block.schemas import Block, Edge
from src.response.schemas import ImmutableBaseModel


class GetCanvasResponse(ImmutableBaseModel):
    blocks: List[Block]
    edges: List[Edge]

class SaveCanvasRequest(ImmutableBaseModel):
    blocks: List[Block]
    edges: List[Edge]