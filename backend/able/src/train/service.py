from train.utils import Trainer
from . import TrainRequestDto

class TrainService:
    def train(self, train_request_dto: TrainRequestDto):
        pass
    
def get_train_service() -> TrainService:
    return TrainService()