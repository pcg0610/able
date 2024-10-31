from fastapi import APIRouter
from src.block.enums import BlockType
from src.block.schemas import BlocksResponse, BlockResponse
from src.block.service import find_blocks_by_type
from src.response.schemas import ResponseModel
from src.response.utils import ok

block_router = router = APIRouter()

@router.get("", response_model=ResponseModel[BlocksResponse])
def blocks_by_type(type: BlockType):
    return ok(
        data=BlocksResponse(
            blocks=find_blocks_by_type(type)
        )
    )

@router.get("/search", response_model=BlockResponse)
def search(keyword: str):
    return ok(
        data=BlockResponse(blocks=search(keyword))
    )