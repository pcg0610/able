import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from . import BlockDto, EdgeDto
from src.block.enums import BlockType

MAX_LOSS = 10e8

class Trainer:
    """모델의 학습을 책임지는 클래스
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: optim.Optimizer, optimizer: nn.Module, save_path: str, checkpoint_interval: int = 10, device: str = 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.chechpoint_interval = checkpoint_interval
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
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

        epoch_loss = running_loss / len(self.val_loader)
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
            if epoch % self.chechpoint_interval == 0:
                torch.save(self.model.state_dict(), f"model_checkpoint_epoch_{epoch+1}.pth")
                
            if best_train_loss > train_loss:
                torch.save(self.model.state_dict(), f"model_checkpoint_best_train_loss.pth")
            
            if best_valid_loss > valid_loss:
                torch.save(self.model.state_dict(), f"model_checkpoint_best_valid_loss.pth")
                
class Logger:
    """학습 과정을 저장하기 위한 클래스
    """
    def __init__(self):
        pass
    
def convert_blockgraph_to_model(blocks: list[BlockDto], edges: list[EdgeDto]) -> nn.Module:
    
    pass

def find_data_block(blocks: list[BlockDto]) -> BlockDto | None:
    for block in blocks:
        if block.type == BlockType.DATA:
            return block
    return None

def find_interpreter_block(blocks: list[BlockDto]) -> BlockDto | None:
    for block in blocks:
        if block.type == BlockType.INTERPRETER:
            return block
    return None

def validate_data_path(data_path: str) -> bool:
    pass

def create_data_loaders(data_path: str) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    pass