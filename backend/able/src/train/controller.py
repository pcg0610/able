from fastapi import APIRouter, Response, Depends, BackgroundTasks
from . import TrainRequestDto, TrainResponseDto
from . import TrainService, get_train_service

train_router = APIRouter(prefix="/api/v1/train")

@train_router.post("", tags=["train"])
async def train(train_request_dto: TrainRequestDto, backgroud_tasks: BackgroundTasks, train_service: TrainService = Depends(get_train_service)) -> TrainResponseDto:
    
    backgroud_tasks.add_task(train_service.train(train_request_dto))
    
    return Response()