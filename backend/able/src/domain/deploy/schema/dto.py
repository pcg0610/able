from src.response.schemas import ImmutableBaseModel

class ApiInformation(ImmutableBaseModel):
    project_name: str
    train_result: str
    checkpoint: str
    uri: str
    description: str
