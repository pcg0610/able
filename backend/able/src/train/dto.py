from src.block.schemas import Block, Edge
from src.response.schemas import ImmutableBaseModel


class TrainRequest(ImmutableBaseModel):
    project_name: str
    epoch: int
    batch_size: int
    block: tuple[Block]
    edges: tuple[Edge]

class TrainResponse(ImmutableBaseModel):
    pass