import json
from typing import Any, Dict
from pydantic import BaseModel
import logging
import base64
import re

from src.file.exceptions import FileNotFoundException

logger = logging.getLogger(__name__)

def str_to_json(data: str) -> Dict[str, Any]:

    if not isinstance(data, str):
        logger.error(f"Invalid data type: Expected a JSON string, but got {type(data)}")
        raise TypeError("Invalid data type: Expected a JSON string.")

    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 디코딩 실패: {e}. 데이터: {data[:50]}...", exc_info=True)
        raise

def json_to_str(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        data_dict = obj.model_dump()
    elif isinstance(obj, dict):
        data_dict = obj
    else:
        raise TypeError("지원되지 않는 데이터 타입입니다. Pydantic 모델 또는 딕셔너리가 필요합니다.")

    try:
        return json.dumps(data_dict, ensure_ascii=False, indent=4)
    except (TypeError, json.JSONDecodeError, UnicodeEncodeError) as e:
        logger.error(f"JSON 직렬화 실패: {e}", exc_info=True)
        raise

def encode_image_to_base64(image_data: bytes) -> str:
    return base64.b64encode(image_data).decode("utf-8")

def get_epoch_id(epoch_dir_name: str) -> int:
    epoch_id = re.search(r'epoch_(\d+)', epoch_dir_name)

    if epoch_id : 
        return int(epoch_id.group(1))
    else :
        raise FileNotFoundException(f"{epoch_dir_name}을 찾을 수 없습니다.")