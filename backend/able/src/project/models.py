from pydantic import BaseModel

class Project(BaseModel):
    title: str                          # 프로젝트명 
    description: str | None = None      # 프로젝트 설명(선택)
    cuda_version: str                   # 쿠다 버전
    python_kernel_path: str             # 파이썬 커널

class SelectedProject(Project):
    thumbnail: str | None = None        # 썸네일

class UpdatedProject(Project):
    prev_title: str                     # 변경 전 프로젝트명
    prev_description: str | None = None # 변경 전 설명