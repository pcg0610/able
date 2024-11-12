import logging

from src.checkpoints.schemas import CheckpointListResponse, CheckpointsPaginatedResponse
from src.file.path_manager import PathManager
from src.file.utils import get_directory
from src.file.constants import *
from src.utils import handle_pagination, has_next_page
from src.checkpoints.utils import sort_checkpoints, get_checkpoints

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pathManager = PathManager()

def get_all_checkpoints(project_name: str, result_name: str) -> CheckpointListResponse:
    checkpoints = get_checkpoints(project_name, result_name)
    checkpoints = [TRAIN_BEST, VALID_BEST, FINAL] + checkpoints
    result = CheckpointListResponse(checkpoints=checkpoints)
    return result

def get_paginated_checkpoints(project_name: str, result_name: str, index: int, size: int) -> CheckpointsPaginatedResponse:
    checkpoints = sort_checkpoints(get_checkpoints(project_name, result_name))
    page_item = handle_pagination(checkpoints, index, size)
    return CheckpointsPaginatedResponse(checkpoints=page_item, has_next=has_next_page(len(checkpoints), index, size))

def search_checkpoint(project_name: str, result_name:str, keyword: str, index: int, size: int) -> CheckpointsPaginatedResponse:
    checkpoints = sort_checkpoints(get_checkpoints(project_name, result_name))
    page_item = handle_pagination(checkpoints, index, size)
    return CheckpointsPaginatedResponse(checkpoints=page_item, has_next=has_next_page(len(checkpoints), index, size))