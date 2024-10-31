from fastapi import APIRouter, HTTPException

from src.canvas.schemas import GetCanvasResponse, SaveCanvasRequest, SaveCanvasResponse
from src.canvas.service import get_block_graph, save_block_graph

canvas_router = router = APIRouter()

@router.get("", response_model=GetCanvasResponse)
def get_canvas(project_name: str):
    data = get_block_graph(project_name)
    return GetCanvasResponse(data=data)

@router.post("")
def save_canvas(project_name: str, data: SaveCanvasRequest):

    success = save_block_graph(project_name, data)

    if success:
        return SaveCanvasResponse(success = success)

