from typing import List, Optional
from src.response.schemas import ImmutableBaseModel

class FeatureMap(ImmutableBaseModel):
    block_id: str
    img: Optional[str]

class CheckpointResponse(ImmutableBaseModel):
    epochs : Optional[List[str]]
    has_next: bool

class ImageResponse(ImmutableBaseModel):
    image: str

class FeatureMapRequest(ImmutableBaseModel):
    project_name: str
    result_name: str
    epoch_name: str
    block_id: List[str]

class FeatureMapResponse(ImmutableBaseModel):
    feature_map: List[FeatureMap]