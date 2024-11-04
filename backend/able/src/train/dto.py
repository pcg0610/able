from src.canvas.schemas import CanvasBlock, Edge
from src.response.schemas import ImmutableBaseModel


class TrainRequest(ImmutableBaseModel):
    project_name: str
    epoch: int
    batch_size: int
    block: tuple[CanvasBlock]
    edges: tuple[Edge]

class TrainResponse(ImmutableBaseModel):
    pass