from starlette import status
from src.exceptions import BaseCustomException


class CanvasNotFoundException(BaseCustomException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Canvas not found or unreadable"
        )