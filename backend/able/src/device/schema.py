from src.response.schemas import ImmutableBaseModel

class Device(ImmutableBaseModel):
    index: int
    name: str