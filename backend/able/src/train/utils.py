from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, Dataset, Subset
from typing import Iterator, Any

from src.block.enums import BlockType
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

from src.block.schemas import Block, Edge
from src.block.utils import convert_block_to_module

from src.file.file_utils import create_file, create_directory
from src.file.path_manager import PathManager
from src.utils import json_to_str
from datetime import datetime

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

MAX_LOSS = 10e8

pathManager = PathManager()

TRAINING_LOSS = "training_loss.json"
VALIDATION_LOSS = "validation_loss.json"
ACCURACY = "accuracy.json"

class TrainLogger:
    def __init__(self, project_name: str, result_name: str = datetime.now().strftime("%Y%m%d_%H%M%S")):
        self.project_name = project_name
        self.result_name = result_name

    def create_epoch_log(self, epoch_id: int, accuracy: float, validation_loss: float, training_loss: float):
        epoch_path = pathManager.get_epoch_path(self.project_name, self.result_name, epoch_id)

        create_file(epoch_path / ACCURACY, json_to_str({'accuracy': accuracy}))
        create_file(epoch_path / VALIDATION_LOSS, json_to_str({'loss': validation_loss}))
        create_file(epoch_path / TRAINING_LOSS, json_to_str({'loss': training_loss}))

    def save_train_result(self, top1_accuracy: float, top5_accuracy: float, precision: float, recall: float, f1: float, fig: Figure):
        # 학습 결과 경로 설정
        result_path = pathManager.get_train_result_path(self.project_name, self.result_name)
        create_directory(result_path)

        # 성능 지표 저장 (performance_metrics.json)
        performance_metrics_data = {
            "metrics": {
                "accuracy": top1_accuracy,
                "top5_accuracy": top5_accuracy,
                "precision": precision,
                "recall": recall
            }
        }
        create_file(result_path / "performance_metrics.json", json_to_str(performance_metrics_data))

        # F1 스코어 저장 (f1_score.json)
        f1_score_data = {"f1_score": f1}
        create_file(result_path / "f1_score.json", json_to_str(f1_score_data))

        # 혼동 행렬 저장 (confusion_matrix.jpg)
        confusion_matrix_path = result_path / "confusion_matrix.jpg"
        fig.savefig(confusion_matrix_path, format="jpg")

class Trainer:
    """모델의 학습을 책임지는 클래스
    """
    def __init__(self, model: nn.Module, dataset: ImageFolder, criterion: nn.Module, optimizer: optim.Optimizer, batch_size, logger: TrainLogger, checkpoint_interval: int = 10, device: str = 'cpu'):
        self.model = model.to(device)
        self.dataset = dataset
        self.train_data_loader, self.valid_data_loader, self.test_data_loader = split_dataset(dataset)
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.checkpoint_interval = checkpoint_interval
        self.device = device

    def train_epoch(self):
        self.model.train()  # 모델을 훈련 모드로 전환
        running_loss = 0.0

        for inputs, targets in self.train_data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Gradients 초기화
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_data_loader)

        return epoch_loss

    def validate(self):
        self.model.eval()  # 모델을 평가 모드로 전환
        running_loss = 0.0

        correct = 0
        total = 0

        with torch.no_grad():  # 평가 시에는 gradients가 필요 없으므로
            for inputs, targets in self.valid_data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)

                _, predicted = torch.max(outputs, 1)  # 예측값 얻기
                correct += (predicted == targets).sum().item()  # 맞춘 샘플 수 누적
                total += targets.size(0)  # 전체 샘플 수 누적

                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

        epoch_loss = running_loss / len(self.valid_data_loader)
        return correct / total, epoch_loss # valid accuracy, loss 반환

    def epoch_accuracy(self):
        self.model.eval()  # 모델을 평가 모드로 전환

        correct = 0  # 정확하게 예측한 샘플 수
        total = 0  # 전체 샘플 수

        with torch.no_grad():  # 평가 시에는 gradients가 필요 없으므로
            for inputs, targets in self.train_data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)

                _, predicted = torch.max(outputs, 1)  # 예측값 얻기
                correct += (predicted == targets).sum().item()  # 맞춘 샘플 수 누적
                total += targets.size(0)  # 전체 샘플 수 누적

        return correct / total # train accuracy 반환

    def train(self, epochs):
        best_train_loss = MAX_LOSS
        best_valid_loss = MAX_LOSS
        
        for epoch in range(epochs):

            # Training and validation
            train_loss = self.train_epoch()
            train_accuracy = self.epoch_accuracy()
            valid_accuracy, valid_loss = self.validate()

            # Logging
            self.logger.create_epoch_log(epoch, train_accuracy, train_loss, valid_loss)

            # Checkpoint (간단히 마지막 모델만 저장)
            if epoch % self.checkpoint_interval == 0:
                torch.save(self.model.state_dict(), f"model_checkpoint_epoch_{epoch+1}.pth")

            if best_train_loss > train_loss:
                torch.save(self.model.state_dict(), f"model_checkpoint_best_train_loss.pth")

            if best_valid_loss > valid_loss:
                torch.save(self.model.state_dict(), f"model_checkpoint_best_valid_loss.pth")

    def test(self):
        self.model.eval()  # 평가 모드로 전환 (드롭아웃 비활성화 등)

        y_true = []
        y_pred = []

        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():  # 그래디언트 비활성화로 메모리 절약
            for inputs, labels in self.test_data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                # Top-1 예측값
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                top1_correct += (predicted == labels).sum().item()  # Top-1 정답 카운트
                total += labels.size(0)

                # Top-5 예측값
                _, top5_pred = outputs.topk(5, dim=1)  # 각 샘플에 대해 상위 5개 예측
                labels = labels.view(-1, 1)  # (batch_size, 1)로 변환
                top5_correct += (top5_pred == labels).sum().item()  # Top-5 정답 카운트

        # 정밀도 계산
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)

        # 재현율 계산
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

        # F1-스코어 계산
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # 혼동 행렬 계산
        fig = plot_confusion_matrix(y_true, y_pred, self.dataset.classes)

        return top1_correct / total, top5_correct / total, precision, recall, f1, fig


def plot_confusion_matrix(y_true, y_pred, class_names):
    # 혼동 행렬 계산
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 플롯 크기 및 스타일 설정
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    # 제목 및 축 레이블 설정
    ax.title("Confusion Matrix")
    ax.xlabel("Predicted Labels")
    ax.ylabel("True Labels")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()

    return fig

def validate_file_format(file_path: str, expected: str) -> bool:
    return file_path.endswith(f".{expected.lower()}")

def create_dataset(data_path: str, train_transform:Compose) -> ImageFolder:
    return ImageFolder(data_path, transform=train_transform)

def create_data_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def split_dataset(dataset: Dataset) -> list[Subset[Any]]:
    return random_split(dataset, [0.6, 0.2, 0.2])

def topological_sort(blocks: tuple[Block], edges: tuple[Edge]) -> tuple[Block]:
    graph = defaultdict(list)
    in_degree = {block.id: 0 for block in blocks}

    for edge in edges:
        graph[edge.source_id].append(edge.target_id)
        in_degree[edge.target_id] += 1

    queue = deque([block for block in blocks if in_degree[block.id] == 0])
    sorted_blocks = []

    while queue:
        block = queue.popleft()
        sorted_blocks.append(block)

        for neighbor in graph[block.id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(next(b for b in blocks if b.id == neighbor))

    if len(sorted_blocks) != len(blocks):
        raise ValueError("Cycle detected in the graph; topological sort not possible.")

    return tuple(sorted_blocks)

class UserModel(nn.Module):
    def __init__(self):
        super(UserModel, self).__init__()
        self.layers = nn.Sequential()

    def forward(self, x):
        return self.forward(x)

def create_data_preprocessor(transform_blocks: tuple[Block]) -> Compose:
    return Compose([convert_block_to_module(transform_block) for transform_block in transform_blocks])

def convert_layer_block_to_module(layer_block: Block) -> nn.Module | None:

    if layer_block.type != BlockType.LAYER:
        return None

    return convert_block_to_module(layer_block)

def convert_criterion_block_to_module(loss_block: Block) -> nn.Module | None:

    if loss_block.type != BlockType.LOSS:
        return None

    return convert_block_to_module(loss_block)

def convert_optimizer_block_to_optimizer(optimizer_block: Block, parameters: Iterator[nn.Parameter]) -> optim.Optimizer | None:

    if optimizer_block.type != BlockType.OPTIMIZER:
        return None

    return convert_block_to_module(optimizer_block, parameters)

def convert_operation_block_to_module(operation_block: Block) -> nn.Module | None:

    if operation_block.type != BlockType.OPERATION:
        return None

    return convert_block_to_module(operation_block)

def convert_block_graph_to_model(blocks: tuple[Block], edges: tuple[Edge]) -> nn.Module | None:
    sorted_blocks = topological_sort(blocks, edges)

    model = UserModel()

    for block in sorted_blocks:
        if block.type == BlockType.LAYER:
            module = convert_layer_block_to_module(block)
        elif block.type == BlockType.OPERATION:
            module = convert_operation_block_to_module(block)
        else:
            return None

        model.layers.add_module(block.name, module)

    return model