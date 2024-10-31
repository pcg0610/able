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

from .schemas import EpochResult
from pathlib import Path
from src.file.file_utils import create_file
from src.file.path_manager import PathManager
from src.utils import json_to_str

MAX_LOSS = 10e8

pathManager = PathManager()

TRAINING_LOSS = "training_loss.json"
VALIDATION_LOSS = "validation_loss.json"
ACCURACY = "accuracy.json"

class TrainLogger:
    """학습 과정을 저장하기 위한 클래스
    """

    def __init__(self):
        pass

class Trainer:
    """모델의 학습을 책임지는 클래스
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, validate_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, logger: TrainLogger, checkpoint_interval: int = 10, device: str = 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.checkpoint_interval = checkpoint_interval
        self.device = device

    def train_epoch(self):
        self.model.train()  # 모델을 훈련 모드로 전환
        running_loss = 0.0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Gradients 초기화
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            print(len((outputs, targets)))

            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate(self):
        self.model.eval()  # 모델을 평가 모드로 전환
        running_loss = 0.0

        with torch.no_grad():  # 평가 시에는 gradients가 필요 없으므로
            for inputs, targets in self.validate_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

        epoch_loss = running_loss / len(self.validate_loader)
        return epoch_loss

    def train(self, epochs):
        best_train_loss = MAX_LOSS
        best_valid_loss = MAX_LOSS
        
        for epoch in range(epochs):

            # Training and validation
            train_loss = self.train_epoch()
            valid_loss = self.validate()

            # Logging
            #TODO: Logging 구현

            # Checkpoint (간단히 마지막 모델만 저장)
            # if epoch % self.checkpoint_interval == 0:
            #     torch.save(self.model.state_dict(), f"model_checkpoint_epoch_{epoch+1}.pth")
            #
            # if best_train_loss > train_loss:
            #     torch.save(self.model.state_dict(), f"model_checkpoint_best_train_loss.pth")
            #
            # if best_valid_loss > valid_loss:
            #     torch.save(self.model.state_dict(), f"model_checkpoint_best_valid_loss.pth")

class Tester:
    pass

def validate_file_format(file_path: str, expected: str) -> bool:
    return file_path.endswith(f".{expected.lower()}")

def create_dataset(data_path: str, train_transform:Compose) -> Dataset:
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

        model.layers.add_module(str(len(model.layers)), module)

    return model

def create_epoch_log(project_name: str, result_name: str, epoch_id: int, epoch_result: EpochResult):
    epoch_path = pathManager.get_epoch_path(project_name, result_name, epoch_id)

    create_file(epoch_path / ACCURACY, json_to_str({'accuracy' : epoch_result.accuracies.accuracy}))
    create_file(epoch_path / VALIDATION_LOSS, json_to_str({'loss' : epoch_result.losses.validation}))
    create_file(epoch_path / TRAINING_LOSS, json_to_str({'loss' : epoch_result.losses.training}))
