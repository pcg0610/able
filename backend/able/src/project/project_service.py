from able.src.project.project_models import Project, SelectedProject, UpdatedProject
from src.file.file_utils import create_directory, get_directory, delete_directory, create_file, get_file, rename_path
from src.file.path_manager import PathManager
from pathlib import Path
from typing import List

path_manager = PathManager()
metadata = "metadata.json"

def create_project(project: Project) -> bool:
    project_path = path_manager.get_projects_path(project.title)
    metadata_path = project_path / metadata
    
    if create_directory(project_path):
        return create_file(metadata_path, project)

    return False


def get_project(title: str) -> SelectedProject:
    metadata_path = path_manager.get_projects_path(title) / metadata

    data = get_file(metadata_path)
    if data is None:
        return None
    
    project = SelectedProject.model_validate(data)

    # 썸네일 가져오기

    return project


def get_projects() -> List[str]:
    projects_path = path_manager.projects_path
    projects = get_directory(projects_path)
    return [project.name for project in projects if project.is_dir()]


def update_project(updated_project: UpdatedProject) -> bool:
    prev_project_path = path_manager.get_projects_path(updated_project.prev_title)
    new_project_path = path_manager.get_projects_path(updated_project.title)
    metadata_path = new_project_path / metadata

    if not (rename_path(prev_project_path, updated_project.title) 
            and updated_project.prev_title != updated_project.title) :
        return False

    project_data = updated_project.model_dump(exclude={"prev_title", "prev_description"})
    new_project = Project(**project_data)

    if updated_project.prev_description != updated_project.description or updated_project.prev_title != updated_project.title:
        return create_file(metadata_path, new_project)

    return True



def delete_project(title: str) -> bool:
    project_path = path_manager.get_projects_path(title)
    return delete_directory(project_path)