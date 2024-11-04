import shutil
import logging
import io
from pathlib import Path
from typing import List
from PIL import Image
from src.file.exceptions import FileNotFoundException, FileUnreadableException

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