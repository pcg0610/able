from src.block.schemas import Block
from src.response.schemas import ImmutableBaseModel

class CanvasBlock(Block):
    block_id: str
    position: str

class Edge(ImmutableBaseModel):
    edge_id: str
    source: str
    target: str

class Canvas(ImmutableBaseModel):
    blocks: list[CanvasBlock] = []
    edges: list[Edge] = []

class GetCanvasResponse(ImmutableBaseModel):
    canvas: Canvas

class SaveCanvasRequest(ImmutableBaseModel):
    canvas: Canvas