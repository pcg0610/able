from pathlib import Path
from src.block.enums import BlockType

HOME_PATH = Path.home()
APPLICATION_NAME = "able"
VERSION = "1.0"

class PathManager:
    
    BASE_PATH = HOME_PATH / APPLICATION_NAME / VERSION

    def __init__(self):
        self.blocks_path = self.BASE_PATH / "blocks"
        self.data_path = self.BASE_PATH / "data"
        self.projects_path = self.data_path / "projects"

    """특정 블록 타입의 디렉터리 경로"""
    def get_block_path(self, block_type: BlockType) -> Path:
        return self.blocks_path / block_type.value

    """프로젝트 기본 경로"""
    def get_project_path(self, name: str) -> Path:
        return self.projects_path / name

    """프로젝트 메타데이터 파일 경로"""
    def get_metadata_path(self, name: str) -> Path:
        return self.get_project_path(name) / "metadata.json"

    """프로젝트 썸네일 이미지 경로"""
    def get_thumbnail_path(self, name: str) -> Path:
        return self.get_project_path(name) / "thumbnail.jpg"

    """프로젝트 블록 그래프 파일 경로"""
    def get_block_graph_path(self, name: str) -> Path:
        return self.get_project_path(name) / "block_graph.json"

    """학습 결과 디렉터리 경로"""
    def get_train_results_path(self, name: str) -> Path:
        return self.get_project_path(name) / "train_results"

    """학습 결과 하위 디렉터리 경로"""
    def get_result_path(self, name: str, result_name: str) -> Path:
        return self.get_train_results_path(name) / result_name

    """에포크별 경로"""
    def get_epoch_path(self, name: str, result_name: str, epoch: int) -> Path:
        return self.get_result_path(name, result_name) / "epochs" / f"epoch_{epoch}"

    """특정 에포크의 피처맵 이미지 디렉터리 경로"""
    def get_feature_maps_path(self, name: str, result_name: str, epoch: int) -> Path:
        return self.get_epoch_path(name, result_name, epoch) / "feature_maps"
