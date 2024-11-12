from src.response.schemas import ImmutableBaseModel

class RegisterApiResponse(ImmutableBaseModel):
    is_success: bool

class RemoveApiResponse(ImmutableBaseModel):
    is_success: bool