from fastapi import APIRouter, Response, BackgroundTasks
from . import TrainRequestDto, TrainResponseDto
from .service import train as train_service

train_router = APIRouter()

@train_router.post("")
async def train(train_request_dto: TrainRequestDto, background_tasks: BackgroundTasks) -> TrainResponseDto:
    
    background_tasks.add_task(train_service(train_request_dto))
    
    return Response()