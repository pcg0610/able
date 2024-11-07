from math import ceil

from src.file.path_manager import PathManager
from src.file.utils import get_file
from src.train_log.utils import parse_train_result_date, format_float
from src.train_log.schemas import TrainLogResponse, TrainSummary
from src.utils import str_to_json, handle_pagination

path_manager = PathManager()
METADATA = "metadata.json"
THUMBNAIL = "thumbnail.jpg"
BLOCK_GRAPH = "block_graph.json"
PERFORMANCE_METRICS = "performance_metrics.json"

def get_train_logs(title:str, page:int, page_size:int) -> list[TrainSummary] :

    train_results_path = path_manager.get_train_results_path(title)
    sorted_train_dirs = sorted(train_results_path.iterdir(), reverse=False)

    train_results = []

    for index, folder_path in enumerate(sorted_train_dirs, start=1):
        metadata_path = folder_path / METADATA
        performance_metrics_path = folder_path / PERFORMANCE_METRICS

        metadata = str_to_json(get_file(metadata_path))
        formatted_date = parse_train_result_date(folder_path.name)

        performance_data = str_to_json(get_file(performance_metrics_path))
        raw_accuracy = performance_data["metrics"].get("accuracy")
        accuracy = format_float(raw_accuracy) if raw_accuracy is not None else "0"

        train_result = TrainSummary(
            index=index,
            date=formatted_date,
            accuracy=accuracy + "%",
            status=metadata["status"]
        )

        train_results.append(train_result)

    train_results.reverse()

    paginated_train_results = handle_pagination(train_results, page, page_size)
    return paginated_train_results or []