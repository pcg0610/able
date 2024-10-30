import logging
from src.block.exceptions import BlockNotFoundException
from src.block.schemas import Block
from src.file.file_utils import get_file, get_directory
from src.file.path_manager import PathManager
from src.utils import str_to_json

logger = logging.getLogger(__name__)
path_manager = PathManager()

def search(
        keyword: str
) -> Block:

    types_dir = get_directory(path_manager.blocks_path)

    for type_dir in types_dir:
        blocks_path = get_directory(type_dir)
        for block_path in blocks_path:
            logger.debug(f"Checking path: {block_path}")
            if block_path.name == f"{keyword}.json":
                block_data = get_file(block_path)
                return Block(**str_to_json(block_data))

    raise BlockNotFoundException(keyword)