import logging

from pathlib import Path
from typing import List
from fastapi import UploadFile

from src.file.path_manager import PathManager
from src.file.utils import get_directory, read_image_file, save_img, create_directory, get_file
from src.utils import encode_image_to_base64, get_epoch_id, str_to_json
from src.analysis.utils import FeatureMapExtractor, read_blocks, preprocess_image, load_model, load_parameter
from src.train.utils import split_blocks
from src.canvas.schemas import Canvas

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
    epoch_path = pathManager.get_epoch_path(project_name, result_name, get_epoch_id(epoch_name))
    feature_maps_path = pathManager.get_feature_maps_path(project_name, result_name, get_epoch_id(epoch_name))

    #block_graph.json 파일에서 블록 읽어오기
    block_graph_path = pathManager.get_train_result_path(project_name, result_name) / "block_graph.json"
    block_graph = read_blocks(block_graph_path)
    blocks = block_graph.blocks 

    # 블록 카테고리 별로 나누기
    _, transform_blocks, _, _, _ = split_blocks(blocks)

    # 피쳐맵을 저장할 디렉터리 생성 및 원본 이미지 저장
    create_directory(feature_maps_path)
    img_path = await save_img(feature_maps_path, "original.jpg", file)

    # 모델 로드
    model_path = pathManager.get_train_result_path(project_name, result_name) / "model.pth"
    model = load_model(model_path)
    
    # 파라미터 적용
    parameter_path = str(epoch_path / "parameter.pth")
    load_parameter(model, parameter_path)

    extractor = FeatureMapExtractor(model, epoch_path, feature_maps_path, transform_blocks, img_path, device="cpu")

    # 피쳐맵 추출 훅 적용 후 실행
    extractor.analyze()

    heatmap_img = encode_image_to_base64(read_image_file(epoch_path / "heatmap.jpg"))
    return heatmap_img

def get_model(project_name: str, result_name: str) -> Canvas :
    block_graph_path = pathManager.get_train_result_path(project_name, result_name) / "block_graph.json"
    block = Canvas(**str_to_json(get_file(block_graph_path)))
    return block