from fastapi import APIRouter

from src.response.schemas import Response
import src.analysis.service as service
from src.response.utils import ok

analysis_router = router = APIRouter()

@router.get("/", response_model=Response,
            summary="epoch 목록 조회", description="")
async def get_epochs(project_name: str, result_name: str):
    epochs = service.get_epochs(project_name, result_name)
    return ok(epochs)

@router.get("/feature-map", response_model=Response,
            summary="피쳐 맵 조회", description="특정 블록의 피쳐맵 조회")
async def get_result(project_name: str, result_name: str, epoch_name:str, block_id: str):
    image = service.get_result(project_name, result_name, epoch_name, block_id)
    return ok(image)