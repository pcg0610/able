from typing import List, Dict, Any
from pydantic import Field
from src.response.schemas import ImmutableBaseModel

class PerformanceMetrics(ImmutableBaseModel):
    accuracy: float
    top5_accuracy: float
    precision: float
    recall: float

class Loss(ImmutableBaseModel):
    training: float
    validation: float

class Accuracy(ImmutableBaseModel):
    accuracy: float

class EpochResult(ImmutableBaseModel):
    epoch: str
    losses: Loss
    accuracies: Accuracy

class TrainResultResponse(ImmutableBaseModel):
    confusion_matrix: str
    performance_metrics: PerformanceMetrics
    f1_score: str
    epoch_result: List[EpochResult]

class TrainResultMetrics(ImmutableBaseModel):
    top1_accuracy: float
    top5_accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: str  # 혼동 행렬 이미지 경로 또는 Base64 인코딩

class TrainResultConfig(ImmutableBaseModel):
    data_path: str = Field(..., description="데이터셋의 경로")
    input_shape: List[int] = Field(..., description="입력 데이터 형상")
    classes: List[str] = Field(..., description="클래스 목록")
    hyper_params: Dict[str, Any] = Field(..., description="하이퍼 파라미터 정보")
    block_graph_info: Dict[str, Any] = Field(..., description="모델 구성 요소 정보")