from .utils import Trainer, validate_file_format, create_dataset, create_data_loader, convert_block_graph_to_model, \
    convert_criterion_block_to_module, TrainLogger, convert_optimizer_block_to_optimizer, split_dataset, \
    create_data_preprocessor, filter_blocks_connected_to_data
from . import TrainRequestDto

def train(train_request_dto: TrainRequestDto):
    data_path = train_request_dto.data.args.get("data_path")

    if not validate_file_format(data_path, "json"):
        raise Exception()

    transform_blocks, loss_blocks, optimizer_blocks, other_blocks = filter_blocks_connected_to_data(
        train_request_dto.data, train_request_dto.transforms, train_request_dto.loss, train_request_dto.optimizer,
        train_request_dto.blocks, train_request_dto.edges)

    #TODO: 학습할 모델 그래프 저장 기능 추가

    transforms = create_data_preprocessor(transform_blocks)

    dataset = create_dataset(data_path, transforms)

    model = convert_block_graph_to_model(other_blocks, train_request_dto.edges)

    if model is None:
        raise Exception()

    criterion = convert_criterion_block_to_module(loss_blocks)

    optimizer = convert_optimizer_block_to_optimizer(optimizer_blocks, model.parameters())

    trainer = Trainer(model, dataset, criterion, optimizer, train_request_dto.batch_size, TrainLogger(train_request_dto.project_name))

    trainer.train(train_request_dto.epoch)

    top1_accuracy, top5_accuracy, precision, recall, f1, fig = trainer.test()

    trainer.logger.save_train_result(top1_accuracy, top5_accuracy, precision, recall, f1, fig)