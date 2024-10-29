import json
import shutil
import logging

from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directory(path: Path) -> bool:
    if not path.exists():
        try:
            path.mkdir(parents=True)
            logger.info(f"디렉터리 생성 성공: {path}")
            return True
        except Exception as e:
            logger.error(f"디렉터리 생성 실패: {e}")
            return False
    return False

def get_directory(path: Path) -> List[Path]:
    if path.exists() and path.is_dir():
        return [item for item in path.iterdir()]
    return []

def delete_directory(path: Path) -> bool:
    if path.exists() and path.is_dir():
        try:
            shutil.rmtree(path)
            logger.info(f"디렉터리 삭제 성공: {path}")
            return True
        except Exception as e:
            logger.error(f"디렉터리 삭제 실패: {e}")
            return False
    return False

def create_file(path: Path, obj: Any) -> bool:

    create_directory(path.parent)

    if isinstance(obj, BaseModel):
        data = obj.model_dump()
    else:
        raise TypeError("객체가 Pydantic BaseModel 인스턴스여야 합니다.")

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"파일 생성 성공: {path}")
        return True
    except (TypeError, json.JSONDecodeError) as e:
        logger.error(f"파일 생성 실패: {e}")
        return False

def get_file(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists() and path.is_file():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"파일 읽기 실패: {e}")
            return None
    return None

def delete_file(path: Path) -> bool:
    if path.exists() and path.is_file():
        try:
            path.unlink()
            logger.info(f"파일 삭제 성공: {path}")
            return True
        except Exception as e:
            logger.error(f"파일 삭제 실패: {e}")
            return False
    return False

def rename_path(path: Path, new_name: str) -> bool:
    if path.exists() and new_name.strip() and path.name != new_name:
        new_path = path.parent / new_name
        try:
            path.rename(new_path)
            logger.info(f"이름 변경 성공: {path} -> {new_path}")
            return True
        except Exception as e:
            logger.error(f"이름 변경 실패: {e}")
            return False

    logger.warning(f"이름 변경 실패: {path} -> {new_name}")
    return False