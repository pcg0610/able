import logging
from typing import TypeVar
from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_204_NO_CONTENT
from src.response.schemas import ResponseModel

logger = logging.getLogger(__name__)
T = TypeVar("T")

def ok(data: T) -> ResponseModel[T]:
    return ResponseModel[T](
            status_code=HTTP_200_OK,
            data=data
        )

def created() -> ResponseModel:
    return ResponseModel(
        status_code=HTTP_201_CREATED
    )

def no_content() -> ResponseModel:
    return ResponseModel(
        status_code=HTTP_204_NO_CONTENT
    )