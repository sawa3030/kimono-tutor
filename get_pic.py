from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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
async def root(data: NumberRequest):
    return {"message": int(data.number)+2}


# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse
# import shutil
# import os

# app = FastAPI()

# # 保存ディレクトリ
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @app.post("/upload")
# async def upload_file(name: str = Form(...), image: UploadFile = File(...)):
#     try:
#         # ファイルの保存パス
#         file_path = os.path.join(UPLOAD_DIR, image.filename)
        
#         # ファイルを保存
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(image.file, buffer)

#         return JSONResponse(content={"message": "Upload successful", "filename": image.filename}, status_code=200)

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)
