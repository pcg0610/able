from fastapi import APIRouter
from src.block.schemas import SearchBlockResponse

block_router = router = APIRouter()

@router.get("", response_model=SearchBlockResponse)
def search(keyword: str):
    return SearchBlockResponse(block = search(keyword))
