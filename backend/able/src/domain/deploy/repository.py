import json
from importlib.metadata import metadata

from pathlib import Path

from src.domain.deploy.enums import ApiStatus
from src.file.utils import get_file, create_file, remove_file, get_files
from src.file.path_manager import PathManager
from src.domain.deploy.schema.dto import ApiInformation
from src.domain.deploy.schema.request import RegisterApiRequest
from src.utils import str_to_json, json_to_str, handle_pagination
from typing import Optional

class DeployRepository:

    def __init__(self):
        self.path_manager = PathManager()
        self.METADATA_PATH = self.path_manager.deploy_path / "metadata.json"
        self.ROUTER_DIR_PATH = Path(__file__).resolve().parent.parent.parent / "deploy_server/src/routers"

    def get_metadata(self) -> dict:
        if not self.METADATA_PATH.exists():
            create_file(self.METADATA_PATH, json.dumps({
                "api_version": "0.33.1",
                "port": "8088",
                "pid": None,
                "status": "stop"
            }))
        return str_to_json(get_file(self.METADATA_PATH))

    def update_metadata(self, data: dict):
        create_file(self.METADATA_PATH, json_to_str(data))

    def get_router_metadata(self, path_name: str) -> dict:
        metadata_path = self.path_manager.get_deploy_path() / f"{path_name}.json"
        return str_to_json(get_file(metadata_path))

    def create_router_metadata(self, path_name: str, data: dict) -> bool:
        metadata_path = self.path_manager.get_deploy_path() / f"{path_name}.json"
        return create_file(metadata_path, json_to_str(data))

    def update_router_metadata(self, path_name: str, status: ApiStatus) -> bool:
        metadata_path = self.path_manager.get_deploy_path() / f"{path_name}.json"
        metadata = self.get_router_metadata(path_name)
        metadata.update({"status": status.value})
        return create_file(metadata_path, json_to_str(metadata))


    def delete_router_metadata(self, path_name: str) -> bool:
        file_path = self.path_manager.get_deploy_path() / f"{path_name}.json"
        return remove_file(file_path)

    def create_router(self, path_name: str, content: str) -> bool:
        file_path = self.ROUTER_DIR_PATH / f"{path_name}.py"
        return create_file(file_path, content)

    def delete_router(self, path_name: str) -> bool:
        file_path = self.path_manager.deploy_path / f"{path_name}.py"
        return remove_file(file_path)

    def get_apis(self, page: int, page_size: int) -> Optional[list[ApiInformation]]:
        deploy_list = [f for f in get_files(self.path_manager.deploy_path) if f != 'metadata.json']
        api_list = handle_pagination(deploy_list, page, page_size)
        api_info_list = []

        for api in api_list:
            file_path = self.path_manager.deploy_path / api
            api_info_list.append(ApiInformation(**str_to_json(get_file(file_path))))

        return api_info_list

