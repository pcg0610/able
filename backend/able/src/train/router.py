from fastapi import APIRouter, BackgroundTasks
from starlette.responses import Response

from . import TrainRequest
from .service import train as train_service, load_train_result
from src.train.schemas import TrainResultResponse
from ..response.utils import accepted

train_router = router = APIRouter()

@router.get("/result", response_model=TrainResultResponse)
def get_train_result(project_name: str, train_result_name: str) -> TrainResultResponse:
    return load_train_result(project_name, train_result_name)


@router.post("")
async def train(request: TrainRequest, background_tasks: BackgroundTasks) -> Response:
    background_tasks.add_task(train_service(request))
    return accepted()
