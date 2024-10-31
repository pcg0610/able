from fastapi import APIRouter

from src.block.enums import BlockType
from src.block.schemas import BlockResponse, BlocksResponse

block_router = router = APIRouter()

@router.get("", response_model=BlocksResponse)
def blocks_by_type(type: BlockType):
    return BlocksResponse(blocks = find_blocks_by_type(type))

@router.get("/search", response_model=BlockResponse)
def search(keyword: str):
    return BlockResponse(block = search(keyword))
