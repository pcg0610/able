import os
import sys
import subprocess
import psutil

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
from src.file.constants import METADATA, MODEL
from src.response.utils import ok
from src.train.schemas import TrainResultMetadata
from src.utils import str_to_json

from src.train.utils import load_transform_pipeline

router = APIRouter()
path_manager = PathManager()

@router.post("{request.uri}")
async def path_name_route(image: str = Body(...)):

    project_name = "{request.project_name}"
    train_result = "{request.train_result}"
    checkpoint = "{request.checkpoint}"

    train_result_metadata_path = path_manager.get_train_result_path(project_name, train_result) / METADATA
    metadata = TrainResultMetadata(**str_to_json(get_file(train_result_metadata_path)))

    # base64를 이미지로 변환 
    image = base64.b64decode(image)
    image = Image.open(io.BytesIO(image))
    image = np.array(image)

    # 전처리 파이프라인 가져오기
    transform_pipeline = load_transform_pipeline(project_name, train_result)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    image: torch.Tensor = transform_pipeline(image)
    image = image.unsqueeze(0).to(device=device)

    model = torch.load(path_manager.get_checkpoint_path(project_name, train_result, checkpoint) / MODEL)

    model.to(device)

    model.eval()
    predicted = model(image)

    top_values, top_indices = predicted.topk(1, dim=1)
    top_values = top_values[0].cpu().detach().numpy()
    top_indices = top_indices[0].cpu().detach().numpy()

    predicted_label = metadata.classes[top_indices[0]]
    max_value = top_values[0]

    return ok(
        data=InferenceResponse(
            label = predicted_label,
            probability = max_value
        )
    )
    """