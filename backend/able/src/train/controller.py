from fastapi import APIRouter, Response, BackgroundTasks
from . import TrainRequestDto, TrainResponseDto

train_router = APIRouter(prefix="/api/v1/train")

@train_router.post("", tags=["train"])
async def train(train_request_dto: TrainRequestDto, background_tasks: BackgroundTasks) -> TrainResponseDto:
    
    background_tasks.add_task(train(train_request_dto))
    
    return Response()