import logging
from typing import Any
from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_204_NO_CONTENT
from src.response.schemas import Response

logger = logging.getLogger(__name__)

def ok(data: Any) -> Response:
    return Response(
            status_code=HTTP_200_OK,
            data=data
        )

def created() -> Response:
    return Response(
        status_code=HTTP_201_CREATED
    )

def no_content() -> Response:
    return Response(
        status_code=HTTP_204_NO_CONTENT
    )