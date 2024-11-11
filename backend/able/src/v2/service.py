from math import ceil
from typing import List, Optional
import logging

from src.project.schemas import Project, SelectedProject, UpdatedProject
from src.file.utils import create_directory, get_directory, delete_directory, create_file, get_file, rename_path, \
    read_image_file
from src.file.path_manager import PathManager
from src.utils import str_to_json, json_to_str, encode_image_to_base64
from src.project.exceptions import ProjectNameAlreadyExistsException
from src.file.exceptions import FileNotFoundException
from src.canvas.schemas import Canvas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path_manager = PathManager()
METADATA = "metadata.json"
THUMBNAIL = "thumbnail.jpg"
BLOCK_GRAPH = "block_graph.json"

def get_project_api_list(title: str) -> Optional[SelectedProject]:
    metadata_path = path_manager.get_projects_path(title) / METADATA
    thumbnail_path = path_manager.get_projects_path(title) / THUMBNAIL

    data = get_file(metadata_path)
    project = SelectedProject.model_validate(str_to_json(data))
    try:
        project.thumbnail = encode_image_to_base64(read_image_file(thumbnail_path))
    except FileNotFoundException as e:
        logger.info(f"썸네일이 존재하지 않음: {project.title}")
        project.thumbnail = None

    return project