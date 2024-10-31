from typing import List
from src.schemas import ImmutableBaseModel

class PerformanceMetrics(ImmutableBaseModel):
    accuracy: str
    top5_accuracy: str
    precision: str
    recall: str

class Loss(ImmutableBaseModel):
    training: str
    validation: str

class Accuracy(ImmutableBaseModel):
    epoch: str
    accuracy: str

class EpochResult(ImmutableBaseModel):
    epoch: str
    performance_metrics: PerformanceMetrics
    losses: Loss
    accuracies: Accuracy

class TrainResultResponse(ImmutableBaseModel):
    confusion_matrix: str
    f1_score: str
    epoch_result: List[EpochResult]