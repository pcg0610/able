import torch
import torch.nn as nn
import numpy as np
import cv2
import logging
from PIL import Image
from typing import List
from pathlib import Path

from src.train.utils import create_data_preprocessor, UserModel
from src.canvas.schemas import CanvasBlock, Canvas
from src.file.utils import get_file
from src.utils import str_to_json
from src.analysis.exceptions import ModelLoadException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureMapExtractor:
    def __init__(self, model: nn.Module, epoch_path: Path, feature_maps_path: Path, transform_blocks: List[CanvasBlock], img_path: Path, device: str = 'cpu') -> None:
        self.epoch_path = epoch_path
        self.device: str = device
        self.model: nn.Module = model

        self.feature_maps_path = feature_maps_path
        self.final_feature_map = None 
        self.output = None  

        self.img_path = img_path
        self.transform_blocks = transform_blocks

    # 피쳐맵, 히트맵 생성
    def analyze(self):
        self.build_model()
        self.model.eval()

        input_img = preprocess_image(self.img_path, self.transform_blocks, self.device)
        input_img.requires_grad = True

        self.output = self.model(input_img)
        self.save_heatmap()

    # 훅 적용
    def build_model(self):
        if self.model is None:
            logger.error(f"모델이 로드되지 않은 상태에서 훅 적용")
            raise ModelLoadException("모델이 로드되지 않았습니다.")

         # 모든 Conv2d 레이어에 피쳐맵 생성 hook 등록
        last_conv_layer_name = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer_name, last_module = name, module
                module.register_forward_hook(self.get_hook_fn(name))

        # 마지막 Conv 레이어에만 히트맵 생성 hook 등록
        if last_conv_layer_name:
            last_module.register_forward_hook(self.get_final_layer_hook())

        self.model.to(self.device)

    # 피쳐맵 생성 훅
    def get_hook_fn(self, layer_name: str) -> callable:
        def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            self.save_feature_map(output.detach(), layer_name)
        return hook_fn
    
    # 히트맵 생성 훅
    def get_final_layer_hook(self):
        def hook_fn(module, input, output):
            self.final_feature_map = output
            self.final_feature_map.retain_grad()
        return hook_fn

    def save_feature_map(self, fmap: torch.Tensor, layer_name: str) -> None:
        fmap = fmap[0, 0].cpu().numpy()
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())
        fmap = np.uint8(255 * fmap)
        fmap = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)

         # 원본 이미지 크기로 확대하되, 도트 모양을 유지
        original_image = cv2.imread(str(self.img_path))
        fmap_resized = cv2.resize(fmap, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(self.feature_maps_path / f"{layer_name}.jpg"), fmap_resized)

    def save_heatmap(self) -> None:
        if self.final_feature_map is None:
            raise ValueError("최종 피쳐 맵이 없습니다. 모델을 통해 이미지를 먼저 예측하세요.")
        if self.output is None:
            raise ValueError("출력값이 없습니다.")

        # 예측 클래스에 대한 그래디언트를 계산
        target_class = self.output.argmax(dim=1).item()  # 예측한 클래스의 인덱스
        self.model.zero_grad()
        self.output[0, target_class].backward()  # 특정 클래스에 대한 그래디언트 계산

        # 마지막 Conv 레이어의 피처맵에 대한 그래디언트를 얻고 weights 계산
        gradients = self.final_feature_map.grad[0]
        weights = torch.mean(gradients, dim=(1, 2))

        # 가중치를 곱하고 모든 채널을 합산하여 히트맵 생성
        weighted_sum = torch.sum(self.final_feature_map[0] * weights.view(-1, 1, 1), dim=0)
        heatmap = weighted_sum.detach().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 원본 이미지와 히트맵을 결합
        original_image = cv2.imread(str(self.img_path))
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        overlay = cv2.addWeighted(original_image, 0.6, heatmap_resized, 0.4, 0)
        cv2.imwrite(str(self.epoch_path / "heatmap.jpg"), overlay)


# json 파일을 읽고 블록 리스트를 반환
def read_blocks(path: Path) -> Canvas:
    return Canvas(**str_to_json(get_file(path)))

# 이미지 전처리
def preprocess_image(img_path: str, transform_blocks: List[CanvasBlock], device: str = 'cpu') -> torch.Tensor:
    img = Image.open(img_path).convert('RGB')
    preprocess = create_data_preprocessor(transform_blocks)
    return preprocess(img).unsqueeze(0).to(device)

# 모델 로드
def load_model(path: Path, device: str = 'cpu') -> nn.Module :
    try:
        model = torch.load(path, map_location=device)
        return model
    except Exception as e :
            logger.error(f"모델 로드 실패: {e}")
            raise ModelLoadException("모델 로드에 실패하였습니다.")

# 파라미터 로드
def load_parameter(model: nn.Module, path: str, device: str = 'cpu') :
    try:
        model.load_state_dict(torch.load(path, map_location=device), strict= False)
    except Exception as e:
        logger.error(f"파라미터 적용 실패: {e}")
        raise ModelLoadException("파라미터 적용에 실패하였습니다.")

