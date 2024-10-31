from fastapi import APIRouter, Response, BackgroundTasks
from . import TrainRequestDto, TrainResponseDto
from .service import train as train_service
from src.train.schemas import TrainResultResponse

train_router = router = APIRouter()

@router.get("/result", response_model=TrainResultResponse)
def get_train_result(project_name: str, train_result_name: str) -> TrainResultResponse:
    return TrainResultResponse()


@router.post("")
async def train(train_request_dto: TrainRequestDto, background_tasks: BackgroundTasks) -> TrainResponseDto:
    background_tasks.add_task(train_service(train_request_dto))
    return Response()
