from src.block.enums import BlockType
from src.block.exceptions import BlockNotFoundException
from src.block.schemas import Block
from src.file.utils import get_file, get_directory
from src.file.path_manager import PathManager
from src.utils import str_to_json, logger
from src.domain.block import repository as block_repository

class BlockService:

    def __init__(self, repository: block_repository):
        self.repository = repository
        self.JSON = ".json"
        self.DATA = "data"

    def find_blocks_by_type(self, block_type: BlockType) -> list[Block]:
        block_paths = self.repository.get_target_block_type_directory_paths(block_type)

        blocks = []
        for block_path in block_paths:

            if block_path.suffix != self.JSON:
                logger.warning(f"Skipping non-JSON file: {block_path}")
                continue

            logger.debug(f"Checking path: {block_path}")
            data = get_file(block_path)
            blocks.append(Block(**str_to_json(data)))

        return blocks

    def search(self, keyword: str) -> Block:
        type_directory_names = self.repository.get_block_type_directories()

        for directory_name in type_directory_names:
            blocks_path = get_directory(directory_name)

            if directory_name.name == self.DATA:
                continue

            for block_path in blocks_path:
                logger.debug(f"Checking path: {block_path}")
                if block_path.name == f"{keyword}.json":
                    block_data = get_file(block_path)
                    return Block(**str_to_json(block_data))

        raise BlockNotFoundException(keyword)