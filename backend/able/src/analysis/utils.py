import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from typing import Dict, List
from pathlib import Path

from src.train.utils import convert_block_graph_to_model, create_data_preprocessor
from src.canvas.schemas import CanvasBlock, Edge, Canvas
from src.file.utils import get_file
from src.utils import str_to_json

class FeatureMapExtractor:
    def __init__(self, epoch_path: Path, feature_maps_path: Path, blocks: List[CanvasBlock], edges: List[Edge], device: str = 'cpu') -> None:
        self.device: str = device
        self.model: nn.Module = self.build_model(blocks, edges)
        
        # parameter_path 설정
        self.parameter_path = str(epoch_path / "parameter.pth")
        self.load_model_weights(self.parameter_path)

        self.feature_maps_path = feature_maps_path
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.final_feature_map = None  # 최종 단계의 피처 맵을 저장할 변수
        self.output = None  # 모델 예측 결과를 저장할 변수

    def build_model(self, blocks: List[CanvasBlock], edges: List[Edge]) -> nn.Module:
        model = convert_block_graph_to_model(blocks, edges)
        
         # 마지막 Conv 레이어에 hook 등록
        last_conv_layer_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer_name, last_module = name, module  # 마지막 Conv 레이어 추적
                module.register_forward_hook(self.get_hook_fn(name))

        # 마지막 Conv 레이어에만 final hook 등록
        if last_conv_layer_name:
            last_module.register_forward_hook(self.get_final_layer_hook())

        return model.to(self.device)

    def get_hook_fn(self, layer_name: str) -> callable:
        def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            self.feature_maps[layer_name] = output
        return hook_fn
    
    def get_final_layer_hook(self):
        def hook_fn(module, input, output):
            self.final_feature_map = output.detach()  # 최종 피처 맵 저장
        return hook_fn

    def load_model_weights(self, parameter_path: str) -> None:
        self.model.load_state_dict(torch.load(parameter_path, map_location=self.device))

    def preprocess_image(self, img_path: str, transform_blocks: tuple[CanvasBlock]) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        preprocess = create_data_preprocessor(transform_blocks)
        return preprocess(img).unsqueeze(0).to(self.device)

    def run_inference(self, img_tensor: torch.Tensor) -> None:
        """이미지를 모델에 입력하여 순전파 수행 및 피처 맵 추출"""
        self.model.eval()
        with torch.no_grad():
            self.output = self.model(img_tensor)  # 예측 결과를 저장

    def save_feature_maps(self) -> None:
        """추출된 피처 맵을 .jpg 파일로 저장"""
        for layer_name, fmap in self.feature_maps.items():
            # 피처 맵의 첫 번째 채널을 사용하여 시각화
            fmap = fmap[0, 0].cpu().numpy()  # 첫 번째 배치, 첫 번째 채널
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())  # 정규화
            fmap = np.uint8(255 * fmap)
            fmap = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
            cv2.imwrite(str(self.feature_maps_path / f"{layer_name}.jpg"), fmap)

    def save_heatmap(self, img_path: Path) -> None:
        """최종 피처 맵과 자동으로 계산한 가중치를 사용해 원본 이미지 위에 히트맵 생성"""
        if self.final_feature_map is None:
            raise ValueError("최종 피처 맵이 없습니다. 모델을 통해 이미지를 먼저 예측하세요.")
        if self.output is None:
            raise ValueError("출력값이 없습니다. `run_inference`를 먼저 실행하세요.")

        # 예측 클래스에 대한 그래디언트를 계산
        target_class = self.output.argmax(dim=1).item()  # 예측한 클래스의 인덱스
        self.model.zero_grad()
        self.output[0, target_class].backward()  # 특정 클래스에 대한 그래디언트 계산

        # 마지막 Conv 레이어의 피처맵에 대한 그래디언트를 얻고 weights 계산
        gradients = self.final_feature_map.grad[0]
        weights = torch.mean(gradients, dim=(1, 2))

        # 가중치를 곱하고 모든 채널을 합산하여 히트맵 생성
        weighted_sum = torch.sum(self.final_feature_map[0] * weights.view(-1, 1, 1), dim=0)
        heatmap = weighted_sum.cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 원본 이미지와 히트맵을 결합
        original_image = cv2.imread(str(img_path))
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        overlay = cv2.addWeighted(original_image, 0.6, heatmap_resized, 0.4, 0)
        cv2.imwrite(str(self.epoch_path / "heatmap.jpg"), overlay)




# json 파일을 읽고 블록 리스트를 반환
def read_blocks(path: Path) -> Canvas:
    return Canvas(**str_to_json(get_file(path)))

