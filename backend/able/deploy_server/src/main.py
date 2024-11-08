import sys
import uvicorn
from fastapi import FastAPI

app = FastAPI()

from deploy_server.src.routers.infer import router as infer_router
app.include_router(infer_router)

from deploy_server.src.routers.model import router as model_router
app.include_router(model_router)

pass



@app.get("/")
async def welcome():
    return {"message": "running"}
