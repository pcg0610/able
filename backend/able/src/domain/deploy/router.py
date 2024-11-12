from fastapi import APIRouter, Depends, Response

from src.domain.deploy.schema.response import RemoveApiResponse
from src.domain.deploy.service import DeployService
from src.domain.deploy.repository import DeployRepository
from src.domain.deploy.schema.request import RegisterApiRequest
from src.response.utils import accepted, ok, no_content

deploy_router = router = APIRouter()

def get_deploy_service() -> DeployService:
    repository = DeployRepository()
    return DeployService(repository=repository)

@router.get("/run")
def run(service: DeployService = Depends(get_deploy_service)) -> Response:
    service.run()
    return accepted()

@router.get("/stop")
def stop(service: DeployService = Depends(get_deploy_service)) -> Response:
    service.stop()
    return accepted()

@router.post("/apis")
def register_api(
    request: RegisterApiRequest,
    service: DeployService = Depends(get_deploy_service)
) -> Response:
    result=RemoveApiResponse(is_success=service.register_api(request))
    return ok(data=result)


@router.delete("/apis")
def remove_router(
        uri: str,
        service: DeployService = Depends(get_deploy_service)
) -> Response:
    result=RemoveApiResponse(is_success=service.remove_api(uri))
    return ok(data=result)

@router.post("/restart")
def restart(
    service: DeployService = Depends(get_deploy_service)
) -> Response:
    service.stop()
    service.run()
    return accepted()

@router.get("/apis")
def get_apis(page: int, page_size: int, service: DeployService = Depends(get_deploy_service)):
    api_list = service.get_apis(page, page_size)

    if len(api_list) == 0:
        return no_content()

    return ok(data=api_list)
