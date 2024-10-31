from src.block.schemas import Block, Edge
from src.response.schemas import ImmutableBaseModel

class Canvas(ImmutableBaseModel):
    blocks: list[Block] = []
    edges: list[Edge] = []

class GetCanvasResponse(ImmutableBaseModel):
    canvas: Canvas

class SaveCanvasRequest(ImmutableBaseModel):
    canvas: Canvas