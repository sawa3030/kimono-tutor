from fastapi import FastAPI, Request, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import shutil
from ultralytics import YOLO
from test import dummy_function

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def handler(request: Request, exc: RequestValidationError):
    print(exc)
    return JSONResponse(content={}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


class NumberRequest(BaseModel):
    number: str


@app.post("/")
async def root(picture: UploadFile = File(...)):
    with open("./uploaded/pic.jpg", "wb+") as buffer:
        shutil.copyfileobj(picture.file, buffer)
    # model = YOLO("weights/best.pt")
    # results = model.predict("./uploaded/pic.jpg", save=True)
    return {"message": dummy_function()}
