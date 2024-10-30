from train.utils import Trainer, find_data_block, find_interpreter_block, validate_data_path, create_data_loaders
from . import TrainRequestDto
class TrainService:
    def train(self, train_request_dto: TrainRequestDto):
        data_block = find_data_block(train_request_dto.blocks)
        interpreter_block = find_interpreter_block(train_request_dto.blocks)
        
        if not data_block or not interpreter_block:
            raise Exception()
        
        data_path = data_block.args.get("data_path")
        
        if not validate_data_path(data_path):
            raise Exception()
        
        train_data_loader, valid_data_loader, test_data_loader = create_data_loaders(data_path)
        
        
        
        
    
def get_train_service() -> TrainService:
    return TrainService()