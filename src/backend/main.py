import logging

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


# Reset log every run (should be changed later)
if not logging.getLogger().handlers:
    log_path = ".log"
    with open(log_path, "w") as f:
        f.write("")
    # Create a log file if it doesn't exist
    logging.basicConfig(
        filename=log_path,  # The log file path
        level=logging.INFO,  # Log level; change to DEBUG for more detailed output
        format="%(asctime)s - %(levelname)s - %(message)s",  # Format for each log message
    )


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
