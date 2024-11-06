import subprocess
import sys
import os
import json
from pathlib import Path

import httpx

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

    process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "deploy_server.src.main:app",
        "--host", "0.0.0.0", "--port", "8088", "--reload"
    ])

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


BASE_DIR = Path(__file__).resolve().parent.parent.parent / "deploy_server/src"
ROUTER_DIR = BASE_DIR / "routers"
MAIN_FILE_PATH = BASE_DIR / "main.py"

def register_api(uri: str):

    path_name = uri.strip("/").replace("/", "_")
    file_path = Path(f"{ROUTER_DIR}/{path_name}.py")

    content = f'''
from fastapi import APIRouter

router = APIRouter()

@router.get("{uri}")
async def {path_name}_route():
    return {json.dumps({"message": "running"})}
'''

    if not create_file(file_path, content):
        return False, f"Failed to create router file for path '{uri}'"

    include_statement = f'from deploy_server.src.routers.{path_name} import router as {path_name}_router\n'
    include_statement += f'app.include_router({path_name}_router)\n'
    include_statement += f'\npass\n'

    try:
        # main.py 파일을 읽고 `pass` 위치 찾기
        with MAIN_FILE_PATH.open("r", encoding="utf-8") as main_file:
            main_content = main_file.read()

        # `pass` 위치에 라우터 포함 코드를 삽입
        if "pass" in main_content:
            main_content = main_content.replace("pass", include_statement)

            # 수정된 내용을 main.py에 다시 쓰기
            with MAIN_FILE_PATH.open("w", encoding="utf-8") as main_file:
                main_file.write(main_content)

            return True, f"Route '{uri}' has been created and registered in main.py."
        else:
            return False, "Marker 'pass' not found in main.py."
    except Exception as e:
        return False, f"Failed to update main.py: {e}"
