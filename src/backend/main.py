"""
Main FastAPI application setup for backend services.

This module defines API routes for user registration, login, file uploads,
chat interactions, PDF handling, and summarization. It also initializes
CORS middleware and Prometheus monitoring.
"""

import logging

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from database.db_functions import *
from utils import config # To start the logger first

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for development (allow all)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logger = logging.getLogger(__name__)

# Enable Prometheus metrics
Instrumentator().instrument(app).expose(app)


@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    """
    Register a new user with username and password.
    """
    return db_register(username, password)


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """
    Log in a user and return authentication data.
    """
    return db_login(username, password)


@app.post("/upload")
async def upload_file(user_id: str = Form(...), file: UploadFile = File(...)):
    """
    Upload a PDF file associated with a specific user.
    """
    return db_upload_file(user_id, file)


@app.get("/pdf/{id}/{num}")
async def pdf(id: str, num: int):
    """
    Retrieve a specific page from a PDF document.
    """
    return db_get_pdf(id, num)


@app.get("/chat/{file_id}")
async def chat(file_id: str):
    """
    Retrieve chat history for a given file ID.
    """
    return db_get_chat(file_id)


@app.post("/chatbot")
async def chatbot(prompt: str = Form(...), file_id: str = Form(...)):
    """
    Send a prompt to the chatbot and return its response.
    """
    return db_chatbot(prompt, file_id)


@app.get("/smr/{file_id}")
async def smr(file_id: str):
    """
    Generate a summary of a PDF document.
    """
    return db_smr(file_id)


# Uncomment if needed to clear stored PDFs
# @app.get("/pdf-clear")
# async def remove_pdf():
#     return db_remove_pdf()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
