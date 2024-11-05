from . import TrainRequest
from src.train.schemas import TrainResultResponse, EpochResult, Loss, Accuracy
from .utils import *
from src.file.path_manager import PathManager
from src.file.utils import validate_file_format
from src.utils import encode_image_to_base64
from src.file.utils import load_json_file
from typing import List

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

    # 학습에서 사용하는 간선 추출
    blocks_connected_to_data: tuple[Block, ...] = (data_block, *transform_blocks, *loss_blocks, *optimizer_blocks, *other_blocks)
    canvas_blocks = convert_canvas_blocks(blocks_connected_to_data)
    edges = filter_edges_from_block_connected_data(blocks_connected_to_data, request.edges)

    transforms = create_data_preprocessor(transform_blocks)
    dataset = create_dataset(data_path, transforms)
    model = convert_block_graph_to_model(other_blocks, edges)

    if model is None:
        raise Exception()

    criterion = convert_criterion_block_to_module(loss_blocks)
    optimizer = convert_optimizer_block_to_optimizer(optimizer_blocks, model.parameters())

    # TrainLogger 초기화 전 디렉터리 생성
    project_name = request.project_name
    result_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 결과 및 에포크 디렉터리 생성
    result_path = path_manager.get_train_results_path(project_name) / "train_results" / result_name
    epochs_path = result_path / "epochs"
    create_directory(result_path)
    create_directory(epochs_path)

    # 데이터셋 메타데이터 저장
    save_metadata(project_name, result_name, data_block)

    # 학습 모델 그래프 저장
    save_result_block_graph(request.project_name, result_name, canvas_blocks, edges)

    # 하이퍼 파라미터 정보 저장 (hyper_parameters.json)
    save_result_hyper_parameter(request.project_name, result_name, request.batch_size, request.epoch)

    save_result_model(project_name, result_name, model)

    trainer = Trainer(model, dataset, criterion, optimizer, request.batch_size, TrainLogger(request.project_name))

    trainer.train(request.epoch)
    trainer.test()


def load_train_result(project_name: str, result_name: str) -> TrainResultResponse:
    # 결과 디렉터리 경로 설정
    result_path = path_manager.get_train_results_path(project_name) / result_name

    # 혼동 행렬 이미지 로드 및 인코딩
    confusion_matrix = encode_image_to_base64(result_path / "confusion_matrix.jpg")

    # 성능 지표 로드
    performance_metrics_data = load_json_file(result_path / "performance_metrics.json")
    performance_metrics = PerformanceMetrics(**performance_metrics_data["metrics"])

    # F1 스코어 로드
    f1_score = load_json_file(result_path / "f1_score.json")["f1_score"]

    # 에포크 결과 로드
    epochs_path = path_manager.get_epochs_path(project_name, result_name)
    epoch_results: List[EpochResult] = []

    for epoch_dir in epochs_path.iterdir():
        if epoch_dir.is_dir():
            epoch_id = epoch_dir.name
            # 각 에포크에 해당하는 파일 경로 설정
            training_loss_path = epoch_dir / "training_loss.json"
            validation_loss_path = epoch_dir / "validation_loss.json"
            accuracy_path = epoch_dir / "accuracy.json"

            # 각 파일에서 필요한 데이터 로드
            training_loss = load_json_file(training_loss_path)["loss"]
            validation_loss = load_json_file(validation_loss_path)["loss"]
            accuracy = load_json_file(accuracy_path)["accuracy"]

            # 에포크 결과 인스턴스 생성
            epoch_result = EpochResult(
                epoch=epoch_id,
                losses=Loss(training=training_loss, validation=validation_loss),
                accuracies=Accuracy(accuracy=accuracy)
            )
            epoch_results.append(epoch_result)

    # TrainResultResponse 반환
    return TrainResultResponse(
        confusion_matrix=confusion_matrix,
        performance_metrics=performance_metrics,
        f1_score=f1_score,
        epoch_result=epoch_results
    )