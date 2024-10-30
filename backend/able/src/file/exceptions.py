from src.exceptions import BaseCustomException
from starlette import status

class FileNotFoundException(BaseCustomException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )

class FileUnreadableException(BaseCustomException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )