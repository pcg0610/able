from fastapi import APIRouter, UploadFile, File, Query

from src.response.schemas import ResponseModel
from src.analysis.schemas import CheckpointResponse, ImageResponse, FeatureMapRequest, FeatureMapResponse
import src.analysis.service as service
from src.canvas.schemas import GetCanvasResponse
from src.response.utils import ok, bad_request

analysis_router = router = APIRouter()

@router.get("", response_model=ResponseModel[CheckpointResponse],
            summary="epoch 목록 조회", description="무한 스크롤")
async def get_checkpoints(project_name: str, result_name: str,
                    index: int = Query(..., description="몇 번째 스크롤인지"),
                    size: int = Query(..., description="스크롤 한 번에 가져올 아이템 개수")):
    response = service.get_checkpoints(project_name, result_name, index, size)
    if response.epochs is None:
        return bad_request()
    return ok(data=response)

@router.post("/feature-map", response_model=ResponseModel[FeatureMapResponse],
            summary="피쳐 맵 조회", description="블록의 피쳐맵 조회, 피쳐맵이 존재하지 않는 블록일 경우 null 반환")
async def get_feature_map( request: FeatureMapRequest):
    feature_map_list = service.get_feature_map(request)
    return ok(data=FeatureMapResponse(feature_map=feature_map_list))

@router.post("", 
             response_model=ResponseModel[ImageResponse],
             summary="분석 실행 및 히트맵 생성", description="특정 학습 결과의 에포크에 대해 샘플 이미지 1장을 받아 실행 후 히트맵을 반환" )
async def analyze(project_name: str, result_name: str, epoch_name:str, device_name: str, device_index: int, file: UploadFile = File(...)):
    # if(file.content_type != "imge/jpeg"):
    #     return bad_request()
    image = await service.analyze(project_name, result_name, epoch_name, device_name, device_index, file)
    return ok(data=ImageResponse(image=image))

@router.get("/model",
             summary="특정 학습 결과의 모델(캔버스) 불러오기", description="분석 페이지 접근 시 보여지는 블록 그래프")
async def get_model(project_name:str, result_name:str):
    canvas = service.get_block_graph(project_name, result_name)
    return ok(data=GetCanvasResponse(canvas=canvas))