import logging
from typing import Optional, Dict, Any

from src.canvas.exceptions import CanvasNotFoundException
from src.file.exceptions import FileNotFoundException
from src.file.file_utils import get_file
from src.file.path_manager import PathManager
from src.utils import str_to_json

logger = logging.getLogger(__name__)
path_manager = PathManager()

def get_block_graph(
        project_name: str
) -> Optional[Dict[str, Any]]:

    block_graph_path = path_manager.get_block_graph_path(project_name)

    try:
        file = get_file(block_graph_path)
    except FileNotFoundException as e:
        raise FileNotFoundException(detail=str(e))


    data = str_to_json(file)
    if data is None:
        logger.error(f"블록 그래프 파일을 찾을 수 없거나 읽을 수 없습니다: {block_graph_path}")
        raise CanvasNotFoundException("block_graph.json 파일을 찾을 수 없거나 읽을 수 없습니다.")

    return data
