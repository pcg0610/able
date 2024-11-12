import os
import sys
import subprocess
import psutil

from src.file.constants import *
from src.domain.deploy import repository as deploy_repository
from src.domain.deploy.enums import DeployStatus
from src.domain.deploy.schema.dto import ApiInformation
from src.domain.deploy.schema.request import RegisterApiRequest
from src.domain.deploy.exceptions import AlreadyRunException, AlreadyStopException, AlreadyExistApiException
from src.utils import logger

class DeployService:

    def __init__(self, repository: deploy_repository):
        self.repository = repository
        from pathlib import Path
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent / "deploy_server/src"
        self.ROUTER_DIR = self.BASE_DIR / "routers"
        self.MAIN_FILE_PATH = self.BASE_DIR / "main.py"

    def run(self) -> bool:
        metadata = self.repository.get_metadata()
        if metadata["status"] == DeployStatus.RUNNING.value:
            raise AlreadyRunException()

        main_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../deploy_server/src/main.py"))
        process = subprocess.Popen([sys.executable, main_py_path])
        metadata.update({"pid": process.pid, "status": DeployStatus.RUNNING.value})
        self.repository.update_metadata(metadata)
        return True

    def stop(self) -> bool:
        metadata = self.repository.get_metadata()
        if metadata["status"] == DeployStatus.STOP.value:
            raise AlreadyStopException()

        pid = metadata.get("pid")
        if pid:
            process = psutil.Process(pid)
            process.terminate()
            process.wait()

        metadata.update({"status": DeployStatus.STOP.value, "pid": None})
        self.repository.update_metadata(metadata)
        return True

    def register_api(self, request: RegisterApiRequest) -> bool:
        path_name = request.uri.strip("/").replace("/", "_")
        file_path = self.ROUTER_DIR / f"{path_name}.py"

        if file_path.exists():
            raise AlreadyExistApiException()

        if not self.repository.create_router_metadata_file(path_name, request):
            raise Exception(f"Failed to create metadata file for '{path_name}'")

        router_content = self.generate_router_content(request)
        if not self.repository.create_router_file(path_name, router_content):
            raise Exception(f"Failed to create router file for '{path_name}'")

        # Update main.py to include the new router
        include_statement = f'from deploy_server.src.routers.{path_name} import router as {path_name}_router\napp.include_router({path_name}_router)\n'
        if not self.update_main_file(include_statement):
            raise Exception("Failed to update main.py with new router.")

        return True

    def remove_api(self, uri: str) -> bool:
        path_name = uri.strip("/").replace("/", "_")
        file_path = self.ROUTER_DIR / f"{path_name}.py"
        json_path = self.repository.path_manager.get_deploy_path() / f"{path_name}.json"

        if not self.repository.delete_metadata_file(json_path):
            raise Exception(f"Failed to delete metadata file '{json_path}'")

        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Deleted router file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting router file '{file_path}': {e}")
                return False

        include_statement = f'from deploy_server.src.routers.{path_name} import router as {path_name}_router\napp.include_router({path_name}_router)\n'
        return self.update_main_file(include_statement, add=False)

    def update_main_file(self, include_statement: str, add: bool = True) -> bool:
        try:
            with self.MAIN_FILE_PATH.open("r", encoding="utf-8") as main_file:
                content = main_file.read()

            if add:
                content = content.replace("pass", include_statement + "pass")
            else:
                content = content.replace(include_statement, "")
            with self.MAIN_FILE_PATH.open("w", encoding="utf-8") as main_file:
                main_file.write(content)

            metadata = self.repository.get_metadata()
            if metadata["status"] == DeployStatus.RUNNING.value:
                self.stop()
                self.run()
            return True
        except Exception as e:
            logger.error(f"Failed to update main.py: {e}")
            return False

    def get_apis(self, page: int, page_size: int) -> list[ApiInformation]:
        return self.repository.get_apis(page, page_size)

    def generate_router_content(self, request: RegisterApiRequest) -> str:
        return f"""import torch
import base64
import io
import numpy as np
from PIL import Image
from fastapi import APIRouter, Body
from deploy_server.src.schemas import InferenceResponse
from src.file.path_manager import PathManager
from src.file.utils import get_file
from src.response.utils import ok
from src.train.schemas import TrainResultMetadata
from src.train.utils import create_data_preprocessor, split_blocks
from src.utils import str_to_json
from src.analysis.utils import read_blocks

router = APIRouter()
path_manager = PathManager()

@router.post("{request.uri}")
async def {request.uri.strip("/").replace("/", "_")}_route(image: str = Body(...)):
    project_name, train_result, checkpoint = "{request.project_name}", "{request.train_result}", "{request.checkpoint}"
    metadata_path = path_manager.get_train_result_path(project_name, train_result) / "{METADATA}"
    metadata = TrainResultMetadata(**str_to_json(get_file(metadata_path)))

    # Load block graph
    block_graph_path = path_manager.get_train_result_path(project_name, train_result) / "{BLOCK_GRAPH}"
    block_graph = read_blocks(block_graph_path)

    # Decode and preprocess image
    image = np.array(Image.open(io.BytesIO(base64.b64decode(image))))
    _, transform_blocks, _, _, _ = split_blocks(block_graph.blocks)
    transforms = create_data_preprocessor(transform_blocks)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    image_tensor = transforms(image).unsqueeze(0).to(device)

    # Load model and make prediction
    model = torch.load(path_manager.get_checkpoint_path(project_name, train_result, checkpoint) / "{MODEL}")
    model.to(device).eval()
    predicted = model(image_tensor)
    top_values, top_indices = predicted.topk(1, dim=1)

    predicted_label = metadata.classes[top_indices[0].item()]
    max_value = top_values[0].item()

    return ok(data=InferenceResponse(label=predicted_label, probability=max_value))
    """