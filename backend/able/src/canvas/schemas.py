from typing import List
from pydantic import BaseModel
from src.block.schemas import Block, Edge

class GetCanvasResponse(BaseModel):
    blocks: List[Block]
    edges: List[Edge]

class SaveCanvasResponse(BaseModel):
    success: bool

class SaveCanvasRequest(BaseModel):
    blocks: List[Block]
    edges: List[Edge]