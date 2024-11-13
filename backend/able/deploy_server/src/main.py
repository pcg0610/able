import asyncio
from contextlib import asynccontextmanager

from pathlib import Path
import uvicorn
from fastapi import FastAPI, WebSocket
import argparse

from starlette.websockets import WebSocketDisconnect, WebSocketState
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

def read_log_file():
    with open(log_file_path, "r") as file:
        logs = file.read()
        return logs

async def send_updates(message):
    # 모든 연결된 WebSocket 클라이언트로 메시지 전송
    for client in connected_clients:
        if client.client_state == WebSocketState.CONNECTED:
            await client.send_text(message)

class LogHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()

    def on_modified(self, event):
        # 로그 파일이 변경될 때 이벤트가 발생
        if event.src_path == str(log_file_path):
            logs = read_log_file()
            asyncio.run(send_updates(logs))

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_handler = LogHandler()
    observer = Observer()
    observer.schedule(log_handler, path=str(log_file_path.parent), recursive=False)
    observer.start()

    yield

connected_clients: list[WebSocket] = []
log_file_path = Path(__file__).parent / "server.log"

if log_file_path.exists():
    with open(log_file_path, "w") as f:
        f.truncate(0)

app = FastAPI(lifespan=lifespan)

pass


@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

@app.get("/")
async def welcome():
    return {"message": "running"}

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": f"{str(log_file_path)}",
            "formatter": "default"
        }
    },
    "loggers": {
        "uvicorn": {
            "level": "INFO",
            "handlers": ["file"],
            "propagate": False
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["file"],
            "propagate": False
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["file"],
            "propagate": False
        }
    }
}

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--port", dest="port", default=8088, type=int)
    parser.add_argument("--log-config", dest="log_config", default=logging_config, type=str)

    args = parser.parse_args()

    uvicorn.run(app=app, host="0.0.0.0", port=args.port, log_config=args.log_config)