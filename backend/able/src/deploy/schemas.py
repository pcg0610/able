from src.response.schemas import ImmutableBaseModel

class RegisterRouterRequest(ImmutableBaseModel):
    uri: str
