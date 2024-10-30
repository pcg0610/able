from fastapi import APIRouter, HTTPException

from src.canvas.exceptions import CanvasNotFoundException
from src.canvas.schemas import BlockGraphResponse
from src.canvas.service import get_block_graph

router = APIRouter()

@router.get("/projects/canvas", response_model=BlockGraphResponse)
def get_canvas(project_name: str):
    try:
        data = get_block_graph(project_name)
        return BlockGraphResponse(data = data)
    except CanvasNotFoundException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)