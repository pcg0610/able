import sys
import uvicorn
from fastapi import FastAPI

app = FastAPI()

pass

@app.get("/")
async def welcome():
    return {"message": "running"}
