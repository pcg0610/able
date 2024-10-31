from typing import List
from src.schemas import ImmutableBaseModel

class PerformanceMetrics(ImmutableBaseModel):
    accuracy: float
    top5_accuracy: float
    precision: float
    recall: float

class Loss(ImmutableBaseModel):
    training: float
    validation: float

class Accuracy(ImmutableBaseModel):
    epoch: str
    accuracy: float

class EpochResult(ImmutableBaseModel):
    epoch: str
    performance_metrics: PerformanceMetrics
    losses: Loss
    accuracies: Accuracy

class TrainResultResponse(ImmutableBaseModel):
    confusion_matrix: str
    f1_score: str
    epoch_result: List[EpochResult]