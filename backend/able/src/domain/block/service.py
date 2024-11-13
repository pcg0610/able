from src.block.enums import BlockType
from src.block.exceptions import BlockNotFoundException
from src.block.schemas import Block
from src.utils import str_to_json, logger
from src.domain.block import repository as block_repository

class BlockService:

    def __init__(self, repository: block_repository):
        self.repository = repository

    """
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
    """

    def find_blocks_by_type(self, block_type: BlockType) -> list[Block]:
        block_paths = self.repository.get_json_block_paths(block_type)

        blocks = []
        for block_path in block_paths:
            logger.debug(f"Loading block from path: {block_path}")
            block_data = self.repository.get_block_data(block_path)
            blocks.append(Block(**block_data))

        return blocks

    def search(self, keyword: str) -> Block:
        try:
            block_data = self.repository.get_block_data_by_keyword(keyword)
            return Block(**block_data)
        except BlockNotFoundException:
            logger.error(f"Block not found for keyword: {keyword}")
            raise BlockNotFoundException(keyword)
