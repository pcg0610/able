from .utils import Trainer, validate_file_format, create_dataset, create_data_loader, convert_block_graph_to_model, \
    convert_criterion_block_to_module, TrainLogger, convert_optimizer_block_to_optimizer, split_dataset, \
    create_data_preprocessor
from . import TrainRequestDto

def train(train_request_dto: TrainRequestDto):
    data_path = train_request_dto.data.args.get("data_path")

    if not validate_file_format(data_path, "json"):
        raise Exception()

    transforms = create_data_preprocessor(train_request_dto.transforms)

    dataset = create_dataset(data_path, transforms)

    model = convert_block_graph_to_model(train_request_dto.blocks, train_request_dto.edges)

    if model is None:
        raise Exception()

    criterion = convert_criterion_block_to_module(train_request_dto.loss)

    optimizer = convert_optimizer_block_to_optimizer(train_request_dto.optimizer, model.parameters())

    trainer = Trainer(model, dataset, criterion, optimizer, train_request_dto.batch_size, TrainLogger(train_request_dto.project_name))

    trainer.train(train_request_dto.epoch)

    top1_accuracy, top5_accuracy, precision, recall, f1, fig = trainer.test()

    trainer.logger.save_train_result(top1_accuracy, top5_accuracy, precision, recall, f1, fig)