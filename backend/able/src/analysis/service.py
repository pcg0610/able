import logging

from pathlib import Path
from typing import List

from src.file.path_manager import PathManager
from src.file.file_utils import get_directory, read_image_file
from src.utils import encode_image_to_base64, get_epoch_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pathManager = PathManager()

def get_result(project_name: str, result_name: str, epoch_name:str, block_id: str) -> str:
    feature_map_path = pathManager.get_feature_maps_path(project_name, result_name, get_epoch_id(epoch_name))
    
    image_name = block_id + ".jpg"
    feature_map_image = encode_image_to_base64(read_image_file(feature_map_path / image_name))

    return feature_map_image