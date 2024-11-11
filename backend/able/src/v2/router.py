import src.deploy.service as service
from fastapi import APIRouter
from starlette.responses import Response
from src.response.utils import accepted, ok, no_content, bad_request
from src.deploy.schemas import RegisterApiRequest

v2_router = router = APIRouter()

@router.get("/api/{project_name}",
            summary="배포된 API 리스트 조회", description="")
def get_api_list(project_name: str, page: int, page_size: int):
    api_list = service.get_api_list(page, page_size)
    if api_list is None:
        return bad_request()
    if len(api_list) == 0:
        return no_content()
    return ok(data=api_list)