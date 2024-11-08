from math import ceil
from typing import List, Optional
import logging

from src.project.schemas import Project, SelectedProject, UpdatedProject
from src.file.utils import create_directory, get_directory, delete_directory, create_file, get_file, rename_path, read_image_file
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

def create_project(project: Project) -> bool:
    project_path = path_manager.get_projects_path(project.title)
    train_results_path = path_manager.get_train_results_path(project.title)

    metadata_path = project_path / METADATA
    block_graph_path = project_path / BLOCK_GRAPH
    
    if create_directory(train_results_path):
        create_file(block_graph_path, json_to_str(Canvas()))
        return create_file(metadata_path, json_to_str(project))

    logger.error(f"동일한 디렉터리 이름 존재: {project.title}")
    raise ProjectNameAlreadyExistsException()


def get_project(title: str) -> Optional[SelectedProject]:
    metadata_path = path_manager.get_projects_path(title) / METADATA
    thumbnail_path = path_manager.get_projects_path(title) / THUMBNAIL

    data = get_file(metadata_path)
    project = SelectedProject.model_validate(str_to_json(data))
    try :
        project.thumbnail = encode_image_to_base64(read_image_file(thumbnail_path))
    except FileNotFoundException as e:
        logger.info(f"썸네일이 존재하지 않음: {project.title}")
        project.thumbnail = None

    return project

def get_projects() -> List[str]:
    projects_path = path_manager.projects_path
    projects = get_directory(projects_path)
    return [project.name for project in projects if project.is_dir()]


def update_project(updated_project: UpdatedProject) -> bool:
    prev_project_path = path_manager.get_projects_path(updated_project.prev_title)
    new_project_path = path_manager.get_projects_path(updated_project.title)
    metadata_path = new_project_path / METADATA

    if not (rename_path(prev_project_path, updated_project.title) 
            and updated_project.prev_title != updated_project.title) :
        return False

    project_data = updated_project.model_dump(exclude={"prev_title", "prev_description"})
    new_project = Project(**project_data)

    if updated_project.prev_description != updated_project.description or updated_project.prev_title != updated_project.title:
        return create_file(metadata_path, json_to_str(new_project))

    return True



def delete_project(title: str) -> bool:
    project_path = path_manager.get_projects_path(title)
    return delete_directory(project_path)