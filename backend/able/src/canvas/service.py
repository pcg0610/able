import logging
from typing import Optional, Dict, Any
from src.canvas.exceptions import CanvasNotFoundException
from src.canvas.schemas import SaveCanvasRequest
from src.file.file_utils import get_file, create_file
from src.file.path_manager import PathManager
from src.utils import str_to_json, json_to_str

logger = logging.getLogger(__name__)
path_manager = PathManager()
block_graph = "block_graph.json"

def get_block_graph(
        project_name: str
) -> Optional[Dict[str, Any]]:

    block_graph_path = path_manager.get_block_graph_path(project_name)
    file = get_file(block_graph_path)
    data = str_to_json(file)

    if data is None:
        logger.error(f"블록 그래프 파일을 찾을 수 없거나 읽을 수 없습니다: {block_graph_path}")
        raise CanvasNotFoundException()

    return data

def save_block_graph(project_name: str, data: SaveCanvasRequest) -> bool:

    project_path = path_manager.get_projects_path(project_name)
    block_graph_path =  project_path / block_graph

    if create_file(block_graph_path, json_to_str(data)):
        return True

    raise