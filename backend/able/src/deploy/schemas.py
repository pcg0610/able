from src.response.schemas import ImmutableBaseModel
from typing import Any

class RegisterApiRequest(ImmutableBaseModel):
    project_name: str
    train_result: str
    checkpoint: str
    uri: str
    description: str
    
class ApiInformation(ImmutableBaseModel):
    project_name: str
    train_result: str
    checkpoint: str
    uri: str
    description: str
    
class InferenceResponse(ImmutableBaseModel):
    label: Any
    probability: float
    