import subprocess
import sys
import re
import json
import time
import psutil
from pathlib import Path
from src.deploy.enums import DeployStatus
from src.file.utils import get_file, create_file
from src.file.path_manager import PathManager
from src.utils import str_to_json, json_to_str

path_manager = PathManager()

DEFAULT_METADATA = """{
    "api_version": "0.33.1",
    "port": "8088",
    "reloader_pid": null,
    "server_pid": null,
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
        "--host", "127.0.0.1", "--port", "8088", "--reload"
    ])

    # reloader 프로세스의 PID
    reloader_pid = process.pid
    server_pid = None

    # 특정 포트를 사용하는 프로세스 찾기 (최대 10초 동안 재시도)
    for _ in range(20):
        time.sleep(0.5)
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == 8088 and conn.status == psutil.CONN_LISTEN:
                server_pid = conn.pid
                break
        if server_pid:
            break

    if server_pid:
        print(f"Reloader PID: {reloader_pid}, Server PID: {server_pid}")
        metadata.update({"reloader_pid": reloader_pid, "server_pid": server_pid, "status": DeployStatus.RUNNING.value})
        create_file(metadata_path, json_to_str(metadata))
    else:
        print("Server process could not be found.")

    return True

def stop() -> bool:
    metadata_path = path_manager.deploy_path / "metadata.json"
    if not metadata_path.exists():
        print("No metadata file found. Server may not be running.")
        return False

    # metadata 파일 읽기
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # 서버가 실행 중이 아닌 경우
    if metadata.get("status") != DeployStatus.RUNNING.value:
        print("Server is not running.")
        return False

    # reloader_pid와 server_pid 확인
    reloader_pid = metadata.get("reloader_pid")
    server_pid = metadata.get("server_pid")

    if not reloader_pid or not server_pid:
        print("Process IDs not found in metadata.")
        return False

    try:
        # reloader 프로세스 종료
        reloader_process = psutil.Process(reloader_pid)
        for child in reloader_process.children(recursive=True):
            try:
                child.terminate()  # 자식 프로세스 종료
                child.wait(timeout=5)
                print(f"Terminated server process with PID {child.pid}.")
            except psutil.NoSuchProcess:
                print(f"Server process with PID {child.pid} already terminated.")
        reloader_process.terminate()  # reloader 프로세스 종료
        reloader_process.wait(timeout=5)
        print(f"Terminated reloader process with PID {reloader_pid}.")

        # metadata 파일 갱신
        metadata.update({"status": DeployStatus.STOP.value, "reloader_pid": None, "server_pid": None})
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        print("Server stopped successfully and metadata updated.")

        return True
    except psutil.NoSuchProcess:
        print("The process does not exist or is already terminated.")
        return False
    except Exception as e:
        print(f"Failed to stop server: {e}")
        return False

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

            return True
        else:
            return False
    except Exception as e:
        return False

def remove_api(uri: str) -> bool:

    path_name = uri.strip("/").replace("/", "_")
    file_path = ROUTER_DIR / f"{path_name}.py"

    # 라우터 파일이 존재하면 삭제
    if file_path.exists():
        try:
            file_path.unlink()
            print(f"Deleted router file: {file_path}")
        except Exception as e:
            return False
    else:
        print(f"Router file '{file_path}' does not exist.")

    include_statement = f'from deploy_server.src.routers.{path_name} import router as {path_name}_router\n'
    include_statement += f'app.include_router({path_name}_router)\n'

    try:
        # main.py 파일을 읽고 삭제할 코드 부분 찾기
        with MAIN_FILE_PATH.open("r", encoding="utf-8") as main_file:
            content = main_file.read()

        # 삭제할 라우터 코드가 main.py에 포함된 경우에만 진행
        if include_statement in content:
            content = content.replace(include_statement, "")

            # 수정된 내용을 main.py에 다시 쓰기
            with MAIN_FILE_PATH.open("w", encoding="utf-8") as main_file:
                main_file.write(content)

            print(f"Removed route registration for '{uri}' from main.py")
            return True
        else:
            print(f"No matching route registration found in main.py for '{uri}'")
            return False
    except Exception as e:
        return False