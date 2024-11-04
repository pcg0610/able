import shutil
import logging
import io
import json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from fastapi import UploadFile
from src.file.exceptions import FileNotFoundException, FileUnreadableException, ImageSaveFailException


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directory(path: Path) -> bool:
    if not path.exists():
        try:
            path.mkdir(parents=True)
            logger.info(f"디렉터리 생성 성공: {path}")
            return True
        except Exception as e:
            logger.error(f"디렉터리 생성 실패: {e}", exc_info=True)
            return False
    return False

def get_directory(path: Path) -> List[Path]:
    if path.exists() and path.is_dir():
        return list(path.iterdir())
    return []

def delete_directory(path: Path) -> bool:
    if path.exists() and path.is_dir():
        try:
            shutil.rmtree(path)
            logger.info(f"디렉터리 삭제 성공: {path}")
            return True
        except Exception as e:
            logger.error(f"디렉터리 삭제 실패: {e}", exc_info=True)
            return False
    return False

def create_file(path: Path, data: str) -> bool:

    create_directory(path.parent)

    try:
        with path.open("w", encoding="utf-8") as f:
            f.write(data)
        logger.info(f"파일 저장 성공: {path}")
        return True
    except TypeError as e:
        logger.error(f"파일 저장 실패: {e}", exc_info=True)
        return False

def get_file(path: Path) -> str:

    if path.exists() and path.is_file():
        try:
            with path.open("r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise FileUnreadableException(f"파일을 읽을 수 없습니다: {path}") from e

    raise FileNotFoundException(f"파일을 찾을 수 없거나 접근할 수 없습니다: {path}")

def read_image_file(path: Path) -> bytes:
    if path.exists() and path.is_file():
        try:
            with path.open("rb") as f:
                image_data = f.read()
                image = Image.open(io.BytesIO(image_data))
                image.verify()
                return image_data
        except Exception as e:
            raise FileUnreadableException(f"파일을 읽을 수 없습니다: {path}") from e

    raise FileNotFoundException(f"파일을 찾을 수 없거나 접근할 수 없습니다: {path}")

def remove_file(path: Path) -> bool:
    if path.exists() and path.is_file():
        try:
            path.unlink()
            logger.info(f"파일 삭제 성공: {path}")
            return True
        except Exception as e:
            logger.error(f"파일 삭제 실패: {e}", exc_info=True)
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
            logger.error(f"이름 변경 실패: {e}", exc_info=True)
            return False

    logger.warning(f"이름 변경 실패: {path} -> {new_name}")
    return False

def validate_file_format(file_path: str, expected: str) -> bool:
    return file_path.endswith(f".{expected.lower()}")

async def save_img(path: Path, file_name: str, file: UploadFile) -> Path:
    img_path = path / file_name
    try:
        # 파일을 original.jpg로 저장
        with open(img_path, "wb") as image_file:
            content = await file.read()
            image_file.write(content)

    except Exception as e:
        logger.error(f"원본 이미지 저장 실패: {img_path}",exc_info=True)
        raise ImageSaveFailException("원본 이미지 저장에 실패하였습니다.")
    
    return img_path


# JSON 파일을 읽고 저장하는 함수 추가
def load_json_file(file_path: Path) -> Dict[str, Any]:
    """JSON 파일을 읽어 Dictionary로 반환"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        raise FileNotFoundException(f"{file_path}을 찾을 수 없습니다.")
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파일 디코딩 오류: {e}")
        raise

def save_json_file(data: Dict[str, Any], file_path: Path) -> None:
    """Dictionary를 JSON 파일로 저장"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logger.info(f"JSON 파일 저장 성공: {file_path}")
    except Exception as e:
        logger.error(f"JSON 파일 저장 오류: {e}")
        raise