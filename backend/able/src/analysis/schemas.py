from typing import List
from src.response.schemas import ImmutableBaseModel

class EpochsResponse(ImmutableBaseModel):
    epochs : List[str]

class ImageResponse(ImmutableBaseModel):
    image: str