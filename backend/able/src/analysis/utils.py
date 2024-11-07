import torch
import torch.nn as nn
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from typing import List
from pathlib import Path

from src.train.utils import create_data_preprocessor, UserModel
from src.canvas.schemas import CanvasBlock, Canvas
from src.file.utils import get_file, create_file
from src.utils import str_to_json, json_to_str
from src.analysis.exceptions import ModelLoadException
from src.analysis.schemas import ClassScore, ClassScores
from src.train.schemas import TrainResultMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureMapExtractor:
    def __init__(self, model: nn.Module,result_path: Path, checkpoint_path: Path, feature_maps_path: Path, transform_blocks: List[CanvasBlock], img_path: Path, device: str = 'cpu') -> None:
        self.result_path = result_path
        self.checkpoint_path = checkpoint_path
        self.device: str = device
        self.model: nn.Module = model

        self.feature_maps_path = feature_maps_path
        self.final_feature_map = None 
        self.output = None  

        self.img_path = img_path
        self.transform_blocks = transform_blocks

    # 피쳐맵, 히트맵 생성
    def analyze(self) -> ClassScores:
        self.build_model()
        self.model.eval()

        input_img = preprocess_image(self.img_path, self.transform_blocks, self.device)
        input_img.requires_grad = True

        self.output = self.model(input_img)
        self.save_heatmap()
        scores = self.save_top_k_scores()
        return scores

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
        # 모든 채널을 평균하여 하나의 피처맵 생성
        fmap = fmap.mean(dim=1).squeeze().cpu().numpy()

        # 정규화
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())

        # 컬러맵 적용 및 저장
        plt.imshow(fmap, cmap='viridis', interpolation='nearest')
        plt.axis('off')
        plt.savefig(str(self.feature_maps_path / f"{layer_name}.jpg"), bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def save_heatmap(self) -> None:
        if self.final_feature_map is None:
            raise ValueError("최종 피쳐 맵이 없습니다. 모델을 통해 이미지를 먼저 예측하세요.")
        if self.output is None:
            raise ValueError("출력값이 없습니다.")
    
        # 예측된 클래스의 인덱스를 가져와 해당 클래스에 대해 역전파 수행
        target_class = self.output.argmax(dim=1).item()
        self.model.zero_grad()
        self.output[0, target_class].backward()
    
        # Grad-CAM 방식으로 채널별 중요도 계산
        gradients = self.final_feature_map.grad[0]  # 마지막 Conv 레이어의 그래디언트
        weights = torch.mean(gradients, dim=(1, 2))  # 채널별로 평균을 취해 중요도를 계산
    
        # 중요도 가중치를 피처 맵에 곱하여 합산
        weighted_sum = torch.sum(self.final_feature_map[0] * weights.view(-1, 1, 1), dim=0)
        heatmap = weighted_sum.detach().cpu().numpy()
    
        # 히트맵 정규화 및 색상 적용
        heatmap = np.maximum(heatmap, 0)  # 0 이하 값 제거
        heatmap /= heatmap.max()  # 0~1 사이로 정규화
        heatmap = np.uint8(255 * heatmap)  # 0~255 값으로 변환
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
        # 원본 이미지 불러오기 및 히트맵 크기 조정
        original_image = cv2.imread(str(self.img_path))
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
        # 히트맵과 원본 이미지 결합 및 저장
        overlay = heatmap_resized * 0.4 + original_image
        cv2.imwrite(str(self.checkpoint_path / "heatmap.jpg"), overlay)
    

    # 상위 k개의 클래스 이름과 점수 반환(100점 만점)
    def save_top_k_scores(self, k: int = 3) -> List[ClassScore]:
        top_values, top_indices = self.output.topk(k, dim=1)
        top_values = top_values[0].cpu().detach().numpy()
        top_indices = top_indices[0].cpu().detach().numpy()

        class_names = get_class_names(self.result_path / "metadata.json")


        # 1을 기준으로 100점 만점으로 변환
        scores = [
            ClassScore(
                class_name=class_names[idx] if class_names else f"Class {idx}",
                class_score=round(score * 100)  # 1.0을 기준으로 100점 만점으로 변환
            )
            for idx, score in zip(top_indices, top_values)
        ]

        # 파일 기록
        # create_file(self.checkpoint_path / "analysis_result.json", json_to_str(ClassScores(class_scores=scores)))

        return scores




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

def get_class_names(path: Path) -> list[str]:
        return TrainResultMetadata(**str_to_json(get_file(path))).classes
    
