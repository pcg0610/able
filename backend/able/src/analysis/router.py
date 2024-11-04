from fastapi import APIRouter

from src.response.schemas import ResponseModel
from src.analysis.schemas import EpochsResponse, ImageResponse
import src.analysis.service as service
from src.canvas.schemas import Canvas
from src.canvas.service import get_canvas
from src.response.utils import ok

analysis_router = router = APIRouter()

@router.get("/", response_model=ResponseModel[EpochsResponse],
            summary="epoch 목록 조회", description="")
async def get_epochs(project_name: str, result_name: str):
    epochs = service.get_epochs(project_name, result_name)
    return ok(data=EpochsResponse(epochs=epochs))

@router.get("/feature-map", response_model=ResponseModel[ImageResponse],
            summary="피쳐 맵 조회", description="특정 블록의 피쳐맵 조회")
async def get_result(project_name: str, result_name: str, epoch_name:str, block_id: str):
    image = service.get_result(project_name, result_name, epoch_name, block_id)
    return ok(data=ImageResponse(image=image))