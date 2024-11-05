import subprocess
import sys
import os
from src.deploy.enums import DeployStatus
from src.file.utils import get_file, create_file
from src.file.path_manager import PathManager
from src.utils import str_to_json, json_to_str

path_manager = PathManager()
DEFAULT_METADATA = """{
    "api_version": "0.33.1",
    "port": "8088",
    "pid": null,
    "status": "stop"
}"""

def run() -> bool:

    metadata_path = path_manager.deploy_path/"metadata.json"
    if not metadata_path.exists():
        create_file(metadata_path, DEFAULT_METADATA)

    metadata = str_to_json(get_file(metadata_path))

    if metadata["status"] == DeployStatus.RUNNING.value:
        # 예외 처리 필요
        pass

    service_dir = os.path.dirname(__file__)
    main_py_path = os.path.abspath(os.path.join(service_dir, "../../deploy_server/src/main.py"))
    process = subprocess.Popen([sys.executable, main_py_path, str("8088")])
    metadata.update({"pid": process.pid, "status": DeployStatus.RUNNING.value})
    create_file(metadata_path, json_to_str(metadata))

    return True


def stop() -> bool:

    metadata_path = path_manager.deploy_path / "metadata.json"
    metadata = str_to_json(get_file(metadata_path))

    if metadata["status"] == DeployStatus.STOP.value:
        # 예외 처리 필요
        pass

    pid = metadata.get("pid")
    if pid:
        try:
            if os.name == 'nt':
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=True)
            else:
                subprocess.run(["kill", "-9", str(pid)], check=True)
        except subprocess.CalledProcessError:
            raise RuntimeError(f"Failed to stop server with PID {pid}.")

    metadata.update({"status": DeployStatus.STOP.value, "pid": None})
    create_file(metadata_path, json_to_str(metadata))

    return True