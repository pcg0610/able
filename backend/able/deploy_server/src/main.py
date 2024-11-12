import asyncio

from pathlib import Path
import uvicorn
from fastapi import FastAPI, WebSocket
import argparse
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

app = FastAPI()

pass


connected_clients = []
log_file_path = Path(__file__).parent / "server.log"
loop = asyncio.get_event_loop()

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

    # uvicorn 서버를 asyncio 이벤트 루프에서 실행
    config = uvicorn.Config(app=app, host="127.0.0.1", port=args.port, log_config=args.log_config)
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())