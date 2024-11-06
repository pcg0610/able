import src.deploy.service as service
from fastapi import APIRouter
from starlette.responses import Response
from src.response.utils import accepted

deploy_router = router = APIRouter()

@router.get("/run")
def run() -> Response:
    service.run()
    return accepted()

@router.get("/stop")
def stop() -> Response:
    service.stop()
    return accepted()

@router.post("/routers")
def register_router(uri: str) -> Response:
    service.register_router(uri)
    return accepted()

@router.delete("/routers")
def remove_router(uri: str) -> Response:
    service.remove_router(uri)
    return accepted()