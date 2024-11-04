from . import TrainRequest
from .utils import *
from src.file.path_manager import PathManager
from src.file.utils import validate_file_format
from src.canvas.schemas import SaveCanvasRequest, Canvas
from src.canvas.service import save_block_graph

path_manager = PathManager()

def train(request: TrainRequest):

    data_block, transform_blocks, loss_blocks, optimizer_blocks, other_blocks = split_blocks(request.blocks)

    if data_block is None:
        raise ValueError("Data block is required but was not found.")

    data_path = data_block.args.get("data_path")

    if not validate_file_format(data_path, "json"):
        raise Exception()

    transform_blocks, loss_blocks, optimizer_blocks, other_blocks = filter_blocks_connected_to_data(
        data_block, transform_blocks, loss_blocks, optimizer_blocks, other_blocks, request.edges
    )

    save_block_graph(request.project_name, SaveCanvasRequest(canvas=Canvas(blocks=request.blocks, edges=request.edges)))

    transforms = create_data_preprocessor(transform_blocks)
    dataset = create_dataset(data_path, transforms)
    model = convert_block_graph_to_model(other_blocks, request.edges)

    if model is None:
        raise Exception()

    criterion = convert_criterion_block_to_module(loss_blocks)
    optimizer = convert_optimizer_block_to_optimizer(optimizer_blocks, model.parameters())

    # TrainLogger 초기화 전 디렉터리 생성
    project_name = request.project_name
    result_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 결과 및 에포크 디렉터리 생성
    result_path = path_manager.get_train_result_path(project_name, result_name)
    epochs_path = result_path / "epochs"
    create_directory(result_path)
    create_directory(epochs_path)

    trainer = Trainer(model, dataset, criterion, optimizer, request.batch_size, TrainLogger(request.project_name))

    trainer.train(request.epoch)
    trainer.test()