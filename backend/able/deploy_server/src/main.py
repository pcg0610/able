import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket
import argparse
import os

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

app = FastAPI()

from deploy_server.src.routers.infer import router as infer_router
app.include_router(infer_router)

from deploy_server.src.routers.model import router as model_router
app.include_router(model_router)

pass

# WebSocket 클라이언트를 저장할 리스트
connected_clients = []

log_file_path = Path(__file__).parent / "server.log"

# asyncio 이벤트 루프 생성
loop = asyncio.get_event_loop()

# 파일 변경 감지 핸들러 설정
class LogHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()

    async def send_updates(self, message):
        # 모든 연결된 WebSocket 클라이언트로 메시지 전송
        for client in connected_clients:
            await client.send_text(message)

    def on_modified(self, event):
        # 로그 파일이 변경될 때 이벤트가 발생
        if event.src_path == str(log_file_path):
            with open(log_file_path, "r") as file:
                logs = file.read()
                # asyncio 이벤트 루프에서 비동기 작업을 실행하도록 설정
                future = asyncio.run_coroutine_threadsafe(self.send_updates(logs), loop)
                future.result()  # 예외 처리

log_handler = LogHandler()
observer = Observer()
observer.schedule(log_handler, path=str(log_file_path.parent), recursive=False)
observer.start()

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connected_clients.remove(websocket)

@app.get("/")
async def welcome():
    return {"message": "running"}

if __name__=="__main__":
    # 현재 파일과 같은 디렉터리에 있는 logging_config.yaml 파일을 절대 경로로 지정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_log_config = os.path.join(current_dir, "logging_config.yaml")
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--port", dest="port", default=8088, type=int)
    parser.add_argument("--log-config", dest="log_config", default=default_log_config, type=str)

    args = parser.parse_args()

    # uvicorn 서버를 asyncio 이벤트 루프에서 실행
    config = uvicorn.Config(app=app, host="127.0.0.1", port=args.port, log_config=args.log_config)
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())