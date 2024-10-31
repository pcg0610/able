import src.project.service as service

from fastapi import APIRouter, status

from src.project.schemas import Project, UpdatedProject
from src.response.schemas import Response
from src.response.utils import created, ok, no_content

project_router = router = APIRouter()

@router.post("", response_model=Response)
async def create_project(project: Project):
    service.create_project(project)
    return created()

@router.get("/{title}", response_model=Response)
async def get_project(title: str):
    project = service.get_project(title)
    return ok(project)

@router.get("/", response_model=Response)
async def get_projects():
    projects = service.get_projects()
    return ok(projects)

@router.put("", response_model=Response)
async def update_project(project: UpdatedProject):
    service.update_project(project)
    return ok(True)

@router.delete("/{title}", response_model=Response)
async def delete_project(title: str):
    service.delete_project(title)
    return no_content()