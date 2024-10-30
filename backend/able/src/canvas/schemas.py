from typing import List
from pydantic import BaseModel
from src.train import BlockDto, EdgeDto

class GetCanvasResponse(BaseModel):
    blocks: List[BlockDto]
    edges: List[EdgeDto]

class SaveCanvasResponse(BaseModel):
    success: bool

class SaveCanvasRequest(BaseModel):
    blocks: List[BlockDto]
    edges: List[EdgeDto]