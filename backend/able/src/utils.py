import json
from typing import Any, Dict, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

def str_to_json(content: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 디코딩 실패: {e}")
        return None

def json_to_str(data: Any) -> str:
    if isinstance(data, BaseModel):
        data_dict = data.model_dump()
    elif isinstance(data, dict):
        data_dict = data
    else:
        raise TypeError("지원되지 않는 데이터 타입입니다. Pydantic 모델 또는 딕셔너리가 필요합니다.")

    try:
        return json.dumps(data_dict, ensure_ascii=False, indent=4)
    except (TypeError, json.JSONDecodeError) as e:
        logger.error(f"JSON 직렬화 실패: {e}")
        raise
