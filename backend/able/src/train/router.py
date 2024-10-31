from fastapi import APIRouter

from src.train.schemas import TrainResultResponse

router = APIRouter()

@router.get("/result", response_model=TrainResultResponse)
def get_train_result(project_name:str, train_result_name: str) -> TrainResultResponse:
    return TrainResultResponse()
    