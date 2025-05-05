import logging

import utils.config
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlite_db.db_functions import *

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development; restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger = logging.getLogger(__name__)


# @app.lifespan("startup")
# async def startup_event():
#     logger.info("Backend started")


# @app.on_event("shutdown")
# async def shutdown_event():
#     logger.info("Backend stopped")


@app.post("/register")
async def register(
    username: str = Form(...),
    password: str = Form(...),
):
    return db_register(username, password)


@app.post("/login")
async def login(
    username: str = Form(...),
    password: str = Form(...),
):
    return db_login(username, password)


@app.post("/upload")
async def upload_file(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    return db_upload_file(user_id, file)


@app.get("/pdf/{id}/{num}")
async def pdf(id: str, num: int):
    return db_get_pdf(id, num)


@app.get("/chat/{file_id}")
async def chat(file_id: str):
    return db_get_chat(file_id)


@app.post("/chatbot")
async def chatbot(prompt: str = Form(...), file_id: str = Form(...)):
    return db_chatbot(prompt, file_id)


@app.get("/smr/{file_id}")
async def smr(file_id: str):
    return db_smr(file_id)


# @app.get("/pdf-clear")
# async def remove_pdf():
#     return db_remove_pdf()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
