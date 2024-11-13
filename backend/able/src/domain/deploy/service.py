import sys
import subprocess
import psutil

from pathlib import Path
from src.domain.deploy import repository as deploy_repository
from src.domain.deploy.enums import DeployStatus, ApiStatus
from src.domain.deploy.schema.dto import ApiInformation
from src.domain.deploy.exceptions import AlreadyRunException, AlreadyStopException, AlreadyExistsApiException
from src.domain.deploy.utils import *
from src.utils import logger

class DeployService:

    def __init__(self, repository: deploy_repository):
        self.repository = repository
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "deploy_server/src"
        self.ROUTER_DIR = self.BASE_DIR / "routers"
        self.MAIN_FILE_PATH = self.BASE_DIR / "main.py"

    def run(self) -> bool:
        metadata = self.repository.get_metadata()

        if metadata.get("pid"):
            raise AlreadyRunException()

        process = subprocess.Popen([sys.executable, str(self.MAIN_FILE_PATH)])
        metadata.update({"pid": process.pid, "status": DeployStatus.RUNNING.value})
        self.repository.update_metadata(metadata)
        return True

    def stop(self) -> bool:
        metadata = self.repository.get_metadata()
        pid = metadata.get("pid")

        if not pid:
            raise AlreadyStopException()

        try:
            self._terminate_process_tree(pid)
            metadata.update({"status": DeployStatus.STOP.value, "pid": None})
            self.repository.update_metadata(metadata)
            logger.info("Server successfully stopped.")
            return True
        except psutil.NoSuchProcess:
            logger.warning(f"No process with PID {pid} found.")
            return False
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            return False

    def _terminate_process_tree(self, pid: int):
        parent_process = psutil.Process(pid)

        for child in parent_process.children(recursive=True):
            child.terminate()

        psutil.wait_procs(parent_process.children(), timeout=5)
        parent_process.terminate()
        parent_process.wait(timeout=5)

    def register_api(self, request: RegisterApiRequest) -> bool:
        path_name = format_path_name(request.uri)

        if self.repository.exists_router(path_name):
            raise AlreadyExistsApiException()

        api_metadata = request.model_dump()
        api_metadata["status"] = ApiStatus.RUNNING.value
        if not self.repository.create_router_metadata(path_name, api_metadata):
            raise Exception(f"Failed to create metadata file for '{path_name}'")

        router_content = generate_router_content(request)
        if not self.repository.create_router(path_name, router_content):
            raise Exception(f"Failed to create router file for '{path_name}'")

        include_statement = generate_include_statement(path_name)
        if not self._update_main_file(include_statement):
            raise Exception("Failed to update main.py with new router.")

        return True

    def stop_api(self, uri: str) -> bool:
        path_name = format_path_name(uri)

        if not self.repository.delete_router(path_name):
            logger.error(f"Failed to delete router file for '{path_name}'")
            return False

        self.repository.update_router_metadata(path_name, ApiStatus.STOP)

        include_statement = generate_include_statement(path_name)
        if not self._update_main_file(include_statement, add=False):
            raise Exception("Failed to update main.py to remove router.")

        return True

    def _update_main_file(self, include_statement: str, add: bool = True) -> bool:
        try:
            metadata = self.repository.get_metadata()
            was_running = metadata["status"] == DeployStatus.RUNNING.value

            if was_running:
                self.stop()

            with self.MAIN_FILE_PATH.open("r", encoding="utf-8") as main_file:
                content = main_file.read()

            if add:
                content = content.replace("pass", include_statement + "pass")
            else:
                if include_statement in content:
                    content = content.replace(include_statement, "")

            with self.MAIN_FILE_PATH.open("w", encoding="utf-8") as main_file:
                main_file.write(content)

            if was_running:
                self.run()

            return True
        except Exception as e:
            logger.error(f"Failed to update main.py: {e}")
            return False

    def remove_api(self, uri: str) -> bool:
        path_name = format_path_name(uri)
        return self.repository.delete_router_metadata(path_name)

    def get_apis(self, page: int, page_size: int) -> list[ApiInformation]:
        return self.repository.get_apis(page, page_size)