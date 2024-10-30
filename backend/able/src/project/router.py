import src.project.service as service

from fastapi import APIRouter, status

from src.project.models import Project, UpdatedProject
from src.schemas import ResponseSchema

router = APIRouter()

@router.post("/create", response_model=ResponseSchema)
async def create_project(project: Project):
    service.create_project(project)
    return ResponseSchema(status_code=status.HTTP_201_CREATED, detail="프로젝트 생성 완료")

@router.get("/{title}", response_model=ResponseSchema)
async def get_project(title: str):
    project = service.get_project(title)
    return ResponseSchema(status_code=status.HTTP_200_OK, detail=f"{title} 조회 성공", data=project)

@router.get("/", response_model=ResponseSchema)
async def get_projects():
    projects = service.get_projects()
    return ResponseSchema(status_code=status.HTTP_200_OK, detail="프로젝트 목록 조회 성공", data=projects)

@router.put("/update", response_model=ResponseSchema)
async def update_project(project: UpdatedProject):
    service.update_project(project)
    return ResponseSchema(status_code=status.HTTP_200_OK, detail="프로젝트 정보 수정 성공")

@router.delete("/delete/{title}", response_model=ResponseSchema)
async def delete_project(title: str):
    service.delete_project(title)
    return ResponseSchema(status_code=status.HTTP_200_OK, detail=f"{title} 삭제 성공")