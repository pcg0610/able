from src.block.schemas import Block
from src.canvas.schemas import Canvas
from src.response.schemas import ImmutableBaseModel


class ValidateCanvasRequest(ImmutableBaseModel):
    canvas: Canvas

class ValidateCanvasResponse(ImmutableBaseModel):
    has_cycle: bool
    cycle_blocks: tuple[Block]