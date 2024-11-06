import src.deploy.service as service
from fastapi import APIRouter
from starlette.responses import Response
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
async def register_router(uri: str) -> Response:
    service.register_api(uri)
    return accepted()

@router.delete("/routers")
async def remove_router(uri: str) -> Response:
    service.remove_api(uri)
    return accepted()