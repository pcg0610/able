from pathlib import Path

from src.block.enums import BlockType
from src.file.path_manager import PathManager
from src.file.utils import get_directory
from src.utils import logger

class BlockRepository:

    def __init__(self):
        self.path_manager = PathManager()

    def get_target_block_type_directory_paths(self, block_type: BlockType) -> list[Path]:
        block_type_dir_path = self.path_manager.get_block_path(block_type)
        logger.info(f"Searching for blocks of type '{block_type.value}' in {block_type_dir_path}")

        return get_directory(block_type_dir_path)

    def get_block_type_directories(self) -> list[Path]:
        return get_directory(self.path_manager.blocks_path)

