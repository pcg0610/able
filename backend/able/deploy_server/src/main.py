import sys
import uvicorn
from fastapi import FastAPI

app = FastAPI()



pass



@app.get("/")
async def welcome():
    return {"message": "running"}

if __name__ == "__main__":
    port = int(sys.argv[1])
    uvicorn.run(app, port=port)
