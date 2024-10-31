from .utils import Trainer, validate_file_format, create_dataset, create_data_loader, convert_block_graph_to_model, \
    convert_criterion_block_to_module, TrainLogger, convert_optimizer_to_optimizer, split_dataset
from . import TrainRequestDto

def train(train_request_dto: TrainRequestDto):
    data_path = train_request_dto.data.args.get("data_path")

    if not validate_file_format(data_path, "json"):
        raise Exception()

    dataset = create_dataset(data_path, None)

    train_dataset, valid_dataset, test_dataset = split_dataset(dataset)

    train_data_loader = create_data_loader(train_dataset, train_request_dto.batch_size)
    valid_data_loader = create_data_loader(valid_dataset, train_request_dto.batch_size)
    test_data_loader = create_data_loader(test_dataset, train_request_dto.batch_size)

    model = convert_block_graph_to_model(train_request_dto.blocks, train_request_dto.edges)

    criterion = convert_criterion_block_to_module(train_request_dto.loss)

    optimizer = convert_optimizer_to_optimizer(train_request_dto.optimizer, model.parameters())

    trainer = Trainer(model, train_data_loader, valid_data_loader, criterion, optimizer, TrainLogger())

    trainer.train(train_request_dto.epoch)