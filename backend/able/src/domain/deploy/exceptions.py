from starlette import status
from src.exceptions import BaseCustomException

class AlreadyRunException(BaseCustomException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"이미 실행중입니다."
        )
        
class AlreadyStopException(BaseCustomException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"실행중이 아닙니다.."
        )

class AlreadyExistsApiException(BaseCustomException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"이미 존재하는 API입니다."
        )