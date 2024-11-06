from fastapi import APIRouter, BackgroundTasks
from starlette.responses import Response

from .dto import TrainResultRequest, TrainRequest
from .service import train as train_service, load_train_result
from src.train.schemas import TrainResultResponse
from src.response.utils import ok
from ..response.utils import accepted

train_router = router = APIRouter()

@router.get(
    path="/result",
    response_model=TrainResultResponse,
    summary="학습 결과 조회",
    description="프로젝트 이름, 학습 결과 이름에 대하여 조회한다."
)
def get_train_result(request: TrainResultRequest):
    return ok(
        data=TrainResultResponse(
            train_result=load_train_result(request)
        )
    )


@router.post("")
async def train(request: TrainRequest, background_tasks: BackgroundTasks) -> Response:
    background_tasks.add_task(train_service(request))
    return accepted()
