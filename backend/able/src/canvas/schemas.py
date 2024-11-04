from src.block.schemas import Block, Edge
from src.response.schemas import ImmutableBaseModel

class CanvasBlock(Block):
    block_id: str
    position: str

class Canvas(ImmutableBaseModel):
    blocks: list[CanvasBlock] = []
    edges: list[CanvasBlock] = []

class GetCanvasResponse(ImmutableBaseModel):
    canvas: Canvas

class SaveCanvasRequest(ImmutableBaseModel):
    canvas: Canvas