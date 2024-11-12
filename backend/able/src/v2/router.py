import src.v2.service as service
from fastapi import APIRouter

from src.response.schemas import ResponseModel
from src.response.utils import ok, no_content, bad_request
from src.deploy.schemas import ApiInformation

v2_router = router = APIRouter()

@router.get("/api/{project_name}", response_model=ResponseModel[ApiInformation],
            summary="현재 프로젝트에 배포된 API 리스트 조회", description="")
def get_project_api_list(project_name: str):
    api_list = service.get_project_api_list(project_name)
    if api_list is None:
        return bad_request()
    if len(api_list) == 0:
        return no_content()
    return ok(
        data=api_list
    )