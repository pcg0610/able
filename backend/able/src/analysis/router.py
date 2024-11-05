from fastapi import APIRouter, UploadFile, File

from src.response.schemas import ResponseModel
from src.analysis.schemas import EpochsResponse, ImageResponse
import src.analysis.service as service
from src.response.utils import ok, bad_request

analysis_router = router = APIRouter()

@router.get("", response_model=ResponseModel[EpochsResponse],
            summary="epoch 목록 조회", description="")
async def get_epochs(project_name: str, result_name: str):
    epochs = service.get_epochs(project_name, result_name)
    return ok(data=EpochsResponse(epochs=epochs))

@router.get("/feature-map", response_model=ResponseModel[ImageResponse],
            summary="피쳐 맵 조회", description="특정 블록의 피쳐맵 조회")
async def get_result(project_name: str, result_name: str, epoch_name:str, block_id: str):
    image = service.get_result(project_name, result_name, epoch_name, block_id)
    return ok(data=ImageResponse(image=image))

@router.post("",
             summary="분석 실행 및 히트맵 생성", description="특정 학습 결과의 에포크에 대해 샘플 이미지 1장을 받아 실행 후 히트맵을 반환" )
async def analysis(project_name: str, result_name: str, epoch_name:str, file: UploadFile = File(...)):
    if(file.content_type != "imge/jpeg"):
        return bad_request()
    image = service.analysis(project_name, result_name, epoch_name, file)
    return ok(data=ImageResponse(image=image))