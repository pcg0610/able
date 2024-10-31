from typing import List
from src.block.schemas import Block, Edge
from src.response.schemas import ImmutableBaseModel

class Canvas(ImmutableBaseModel):
    blocks: List[Block] = []
    edges: List[Edge] = []

class GetCanvasResponse(ImmutableBaseModel):
    canvas: Canvas

class SaveCanvasRequest(ImmutableBaseModel):
    canvas: Canvas