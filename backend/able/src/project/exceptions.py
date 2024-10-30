from fastapi import status
from src.exceptions import BaseCustomException


class ProjectNameAlreadyExistsException(BaseCustomException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail="동일한 이름을 가진 프로젝트가 이미 존재합니다."
        )