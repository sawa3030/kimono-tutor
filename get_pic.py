from fastapi import FastAPI, Request, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.exception_handler(RequestValidationError)
async def handler(request:Request, exc:RequestValidationError):
    print(exc)
    return JSONResponse(content={}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

class NumberRequest(BaseModel):
    number: str

@app.post("/")
async def root(picture: UploadFile = File(...)):
    with open("/home/eri/kimono/uploaded/pic.jpg", 'wb+') as buffer:
        shutil.copyfileobj(picture.file, buffer)
    return {"message": 3}
    # return {"message": "Hello"}


