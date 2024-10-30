from .utils import Trainer, find_data_block, find_interpreter_block, validate_data_path, create_data_loaders, \
    convert_block_graph_to_model, find_loss_block, find_optimizer_block
from . import TrainRequestDto

def train(train_request_dto: TrainRequestDto):
    data_path = train_request_dto.data.args.get("data_path")

    if not validate_data_path(data_path):
        raise Exception()

    train_data_loader, valid_data_loader, test_data_loader = create_data_loaders(data_path)

    convert_block_graph_to_model(train_request_dto.blocks, train_request_dto.edges)