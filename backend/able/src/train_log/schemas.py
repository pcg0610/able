from src.response.schemas import ImmutableBaseModel

class TrainSummary(ImmutableBaseModel):
    index: int
    date: str
    accuracy: str
    status: str

class TrainLogResponse(ImmutableBaseModel):
    total_train_logs: int
    train_summaries: list[TrainSummary]