from fastapi import APIRouter, HTTPException

from src.canvas.schemas import GetCanvasResponse, SaveCanvasRequest
from src.canvas.service import get_block_graph, save_block_graph
from src.response.utils import ok, created

canvas_router = router = APIRouter()

@router.get("", response_model=GetCanvasResponse)
def get_canvas(project_name: str):
    return GetCanvasResponse(data=get_block_graph(project_name))

@router.post("")
def save_canvas(project_name: str, data: SaveCanvasRequest):
    save_block_graph(project_name, data)
    return created()

