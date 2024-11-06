import src.deploy.service as service
from fastapi import APIRouter
from starlette.responses import Response

from src.deploy.schemas import RegisterRouterRequest
from src.response.utils import accepted

deploy_router = router = APIRouter()

@router.get("/run")
async def run() -> Response:
    service.run()
    return accepted()

@router.get("/stop")
async def stop() -> Response:
    service.stop()
    return accepted()

@router.post("/routers")
async def register_router(request: RegisterRouterRequest) -> Response:
    service.register_api(request.uri)
    return accepted()