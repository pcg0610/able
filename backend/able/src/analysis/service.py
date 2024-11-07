import logging

from pathlib import Path
from typing import List
from fastapi import UploadFile

from src.file.path_manager import PathManager
from src.file.utils import get_directory, read_image_file, save_img, create_directory, get_file
from src.utils import encode_image_to_base64, get_epoch_id, str_to_json, handle_pagination, has_next_page
from src.analysis.utils import FeatureMapExtractor, read_blocks, load_model, load_parameter
from src.train.utils import split_blocks
from src.canvas.schemas import Canvas
from src.analysis.schemas import FeatureMapRequest, FeatureMap, EpochsResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pathManager = PathManager()

def get_epochs(project_name: str, result_name: str, index: int, size: int) -> EpochsResponse:
    epochs_path = pathManager.get_epochs_path(project_name, result_name)
    epochs = get_directory(epochs_path)

    page_item = handle_pagination(
        [epoch.name for epoch in epochs if epoch.is_dir()],
        index,
        size
    )

    return EpochsResponse(epochs=page_item, has_next=has_next_page(len(epochs), index, size))


def get_feature_map(request: FeatureMapRequest) -> List[FeatureMap]:
    feature_map_path = pathManager.get_feature_maps_path(request.project_name, request.result_name, get_epoch_id(request.epoch_name))
    
    feature_map_list = []

    for id in request.block_id :
        image_name = 'layers.' + id + ".jpg"
        image_data = None
        try:
            image_data = encode_image_to_base64(read_image_file(feature_map_path / image_name))
        except Exception as e:
            logger.error(f"feature map이 존재하지 않는 블록: {id}")
        feature_map_list.append(FeatureMap(block_id=id, img=image_data))

    return feature_map_list

async def analyze(project_name: str, result_name: str, epoch_name:str, file: UploadFile) -> str:
    epoch_path = pathManager.get_epoch_path(project_name, result_name, get_epoch_id(epoch_name))
    feature_maps_path = pathManager.get_feature_maps_path(project_name, result_name, get_epoch_id(epoch_name))

    #block_graph.json 파일에서 블록 읽어오기
    block_graph_path = pathManager.get_train_result_path(project_name, result_name) / "block_graph.json"
    block_graph = read_blocks(block_graph_path)

    # 블록 카테고리 별로 나누기
    _, transform_blocks, _, _, _ = split_blocks(block_graph.blocks)

    # 피쳐맵을 저장할 디렉터리 생성 및 원본 이미지 저장
    create_directory(feature_maps_path)
    img_path = await save_img(feature_maps_path, "original.jpg", file)

    # 모델 로드
    model_path = epoch_path / "model.pth"
    model = load_model(model_path)


    extractor = FeatureMapExtractor(model, epoch_path, feature_maps_path, transform_blocks, img_path, device="cpu")

    # 피쳐맵 추출 훅 적용 후 실행
    extractor.analyze()

    heatmap_img = encode_image_to_base64(read_image_file(epoch_path / "heatmap.jpg"))
    return heatmap_img

def get_model(project_name: str, result_name: str) -> Canvas :
    block_graph_path = pathManager.get_train_result_path(project_name, result_name) / "block_graph.json"
    block = Canvas(**str_to_json(get_file(block_graph_path)))
    return block