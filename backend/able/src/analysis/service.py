import logging

from pathlib import Path
from typing import List
from fastapi import UploadFile

from src.file.path_manager import PathManager
from src.file.utils import get_directory, read_image_file, save_img, create_directory
from src.utils import encode_image_to_base64, get_epoch_id
from src.analysis.utils import FeatureMapExtractor, read_blocks
from src.train.utils import split_blocks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pathManager = PathManager()

def get_epochs(project_name: str, result_name: str) -> List[str]:
    epochs_path = pathManager.get_epochs_path(project_name, result_name)
    epochs = get_directory(epochs_path)
    return [epoch.name for epoch in epochs if epoch.is_dir()]


def get_result(project_name: str, result_name: str, epoch_name:str, block_id: str) -> str:
    feature_map_path = pathManager.get_feature_maps_path(project_name, result_name, get_epoch_id(epoch_name))
    
    image_name = block_id + ".jpg"
    feature_map_image = encode_image_to_base64(read_image_file(feature_map_path / image_name))

    return feature_map_image

async def analysis(project_name: str, result_name: str, epoch_name:str, file: UploadFile) -> str:

    #block_graph.json 파일에서 블록, 엣지 정보 읽어오기
    block_graph_path = pathManager.get_train_result_path(project_name, result_name) / "block_graph.json"
    block_graph = read_blocks(block_graph_path)
    blocks = block_graph.blocks
    edges = block_graph.edges    

    # 블록 카테고리 별로 나누기
    _, transform_blocks, _, _, other_blocks = split_blocks(blocks)

    epoch_path = pathManager.get_epoch_path(project_name, result_name, get_epoch_id(epoch_name))
    feature_maps_path = pathManager.get_feature_maps_path(project_name, result_name, get_epoch_id(epoch_name))

    # 피쳐맵을 저장할 디렉터리 생성
    create_directory(feature_maps_path)

    # 원본 이미지 저장
    img_path = await save_img(feature_maps_path, "original.jpg", file)

    extractor = FeatureMapExtractor(epoch_path, feature_maps_path, other_blocks, edges, device="cpu")
    img_tensor = extractor.preprocess_image(img_path, transform_blocks=transform_blocks)
    extractor.run_inference(img_tensor)
    extractor.save_feature_maps(img_path)
    extractor.save_heatmap(img_path)


    heatmap_img = encode_image_to_base64(read_image_file(img_path))
    return heatmap_img
