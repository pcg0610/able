from pathlib import Path
from math import ceil

from src.block.enums import BlockType
from src.block.exceptions import BlockNotFoundException
from src.file.path_manager import PathManager
from src.file.utils import get_directory, get_file, remove_file, create_file
from src.utils import logger, str_to_json, handle_pagination


class BlockRepository:

    def __init__(self):
        self.path_manager = PathManager()
        self.JSON_TEXT = ".json"
        self.DATA_TEXT = "data"

    # ------ Block Retrieval ------
    def get_json_block_paths(self, block_type: BlockType) -> list[Path]:
        block_type_dir_path = self.get_block_type_path(block_type)
        logger.info(f"Searching for blocks of type '{block_type.value}' in {block_type_dir_path}")

        return [path for path in get_directory(block_type_dir_path) if path.suffix == self.JSON_TEXT]

    def get_block_data_by_keyword(self, keyword: str) -> dict:
        for directory_name in self.get_block_type_directories():
            if directory_name.name == self.DATA_TEXT:
                continue

            blocks_path = get_directory(directory_name)
            for block_path in blocks_path:
                if block_path.name == f"{keyword}{self.JSON_TEXT}":
                    return self.get_block_data(block_path)

        raise BlockNotFoundException(keyword)

    # ------ Helper Methods ------
    def get_block_type_path(self, block_type: BlockType) -> Path:
        return self.path_manager.get_block_path(block_type)

    def get_block_data(self, block_path: Path) -> dict:
        data = get_file(block_path)
        return str_to_json(data)

    def get_block_type_directories(self) -> list[Path]:
        return get_directory(self.path_manager.blocks_path)
