from src.domain.deploy.schema.dto import ApiInformation
from src.response.schemas import ImmutableBaseModel

class RegisterApiResponse(ImmutableBaseModel):
    is_success: bool

class StopApiResponse(ImmutableBaseModel):
    is_success: bool

class RemoveApiResponse(ImmutableBaseModel):
    is_success: bool

class GetApisResponse(ImmutableBaseModel):
    total_pages: int
    apis: list[ApiInformation]

