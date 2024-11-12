import logging

from src.checkpoints.schemas import CheckpointListResponse, CheckpointsPaginatedResponse
from src.file.path_manager import PathManager
from src.file.utils import get_directory
from src.file.constants import *
from src.utils import handle_pagination, has_next_page

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pathManager = PathManager()

def get_checkpoints(project_name: str, result_name: str) -> CheckpointListResponse:
    checkpoints_path = pathManager.get_checkpoints_path(project_name, result_name)
    checkpoints = get_directory(checkpoints_path)

    paths = [epoch.name for epoch in checkpoints]

    result = CheckpointListResponse(checkpoints=paths)
    return result

def get_paginated_checkpoints(project_name: str, result_name: str, index: int, size: int) -> CheckpointsPaginatedResponse:
    checkpoints_path = pathManager.get_checkpoints_path(project_name, result_name)
    checkpoints = get_directory(checkpoints_path)
    items = [epoch.name for epoch in checkpoints if epoch.is_dir() and epoch.name not in {TRAIN_BEST, VALID_BEST, FINAL}]

    items = sorted(items, key=lambda x: int(x.split('_')[1]))
    page_item = handle_pagination(items, index, size)
    return CheckpointsPaginatedResponse(checkpoints=page_item, has_next=has_next_page(len(items), index, size))

def search_checkpoint(project_name: str, result_name:str, keyword: str, index: int, size: int) -> CheckpointsPaginatedResponse:
    checkpoints_path = pathManager.get_checkpoints_path(project_name, result_name)
    checkpoints = get_directory(checkpoints_path)

    items = [checkpoint.name for checkpoint in checkpoints if keyword in checkpoint.name and checkpoint.name not in {TRAIN_BEST, VALID_BEST, FINAL}]
    items = sorted(items, key=lambda x: int(x.split('_')[1]))
    page_item = handle_pagination(items, index, size)
    return CheckpointsPaginatedResponse(checkpoints=page_item, has_next=has_next_page(len(items), index, size))