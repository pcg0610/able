from typing import List, Optional
import logging

from src.deploy.schemas import ApiInformation
from src.file.constants import METADATA
from src.file.utils import get_file, get_files
from src.file.path_manager import PathManager
from src.utils import str_to_json, encode_image_to_base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path_manager = PathManager()

PROJECT_NAME = "project_name"

def get_project_api_list(title: str) -> list[ApiInformation]:
    deploy_path = path_manager.get_deploy_path()
    api_list = [file_name for file_name in get_files(deploy_path) if file_name != METADATA]
    api_info_list = []
    for api in api_list:
        file_path = deploy_path / api
        file_data = str_to_json(get_file(file_path))

        if file_data[PROJECT_NAME] == title:
            api_info_list.append(ApiInformation(**file_data))

    return api_info_list