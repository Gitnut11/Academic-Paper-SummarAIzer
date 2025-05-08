import os
import logging

from fastapi import UploadFile
from fastapi.responses import JSONResponse, Response
from models.qna import PDFProcessor, clear_pdf, qna
from models.summarize import summarize
from database.database import Database
from utils.pdf_utils import extract_page_as_binary, get_sections
from utils.prompt import HELPER
from utils.safety_check import safety_check

# Set up logger for this module
logger = logging.getLogger(__name__)


def db_register(username: str, password: str) -> dict | JSONResponse:
    """
    Registers a new user with the given username and password.

    Args:
        username (str): Username for the new account.
        password (str): Password for the new account.

    Returns:
        dict | JSONResponse: Success message or error response if user already exists.
    """
    with Database() as db:
        logger.info(f"Attempting to register user: {username}")
        if db.create_user(username, password):
            return {"message": "User registered successfully"}
        logger.warning(f"Registration failed: Username '{username}' already exists")
        return JSONResponse(
            content={"error": "Username already exists"},
            status_code=400,
        )


def db_login(username: str, password: str) -> dict | JSONResponse:
    """
    Authenticates a user and retrieves their uploaded files.

    Args:
        username (str): Username of the account.
        password (str): Password of the account.

    Returns:
        dict | JSONResponse: User ID and files if login succeeds, otherwise an error.
    """
    with Database() as db:
        logger.info(f"User login attempt: {username}")
        user_id = db.login(username, password)
        if user_id:
            file_list = db.get_user_files(user_id)
            return {
                "user_id": user_id,
                "files": file_list,
            }
        logger.warning("Login failed: Invalid username or password")
        return JSONResponse(
            content={"error": "Invalid username or password"},
            status_code=401,
        )


def db_upload_file(user_id: str, file: UploadFile):
    """
    Uploads a PDF file for the user and updates the database.

    Args:
        user_id (str): ID of the user uploading the file.
        file (UploadFile): PDF file to upload.

    Returns:
        list | JSONResponse: Updated file list or error message.
    """
    with Database() as db:
        try:
            logger.info(f"Uploading file for user {user_id}...")
            file_binary = file.file.read()
            db.insert_user_file_binary(user_id, file_binary, file.filename)
            file_list = db.get_user_files(user_id)
            return file_list
        except Exception as e:
            logger.error(f"Error during file upload: {str(e)}")
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )


def db_get_pdf(file_id: str, num: int):
    """
    Returns a specific page from a PDF file as binary response.

    Args:
        file_id (str): ID of the PDF file.
        num (int): Page number to extract.

    Returns:
        Response | JSONResponse: PDF page content or error message.
    """
    with Database() as db:
        try:
            logger.info(f"Fetching page {num} from file ID: {file_id}")
            file_path, file_name = db.get_file_by_id(file_id)
            if not os.path.exists(file_path):
                logger.warning(f"PDF file not found at: {file_path}")
                return JSONResponse(
                    content={"error": "File not found"}, status_code=400
                )
            binary = extract_page_as_binary(file_path, num)
            return Response(
                content=binary,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename={file_name}_{num}.pdf"
                },
            )
        except Exception as e:
            logger.error(f"Failed to fetch PDF page: {str(e)}")
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )


def db_get_chat(file_id: str):
    """
    Retrieves chat history associated with a given file ID.

    Args:
        file_id (str): ID of the PDF file.

    Returns:
        JSONResponse: Chat history or error.
    """
    with Database() as db:
        try:
            logger.info(f"Retrieving chat history for file: {file_id}")
            chat_hist = db.get_history(file_id)
            return JSONResponse(content={"history": chat_hist})
        except Exception as e:
            logger.error(f"Failed to load chat history: {str(e)}")
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )


def db_chatbot(prompt: str, file_id: str):
    """
    Processes chatbot commands and responses for a given file.

    Args:
        prompt (str): User input or command.
        file_id (str): ID of the PDF file.

    Returns:
        dict | JSONResponse: Response message or error.
    """
    with Database() as db:
        try:
            db.log_chat(file_id, prompt, role="user")
            prompt = prompt.strip()
            command = prompt.split()[0]

            logger.info(f"Received prompt: {prompt}")

            if command == "/help":
                reply = HELPER
            elif command == "/summarize":
                file = db.get_file_by_id(file_id)
                pdf_process = PDFProcessor()
                text = pdf_process.extract_text(file[0])
                reply = summarize(text)
            elif command == "/sections":
                file = db.get_file_by_id(file_id)
                sections = get_sections(file[0])
                reply = "\n\n".join(sections)
            elif command.startswith("/"):
                reply = "Unknown command. Use `/help` for available commands."
            else:
                # Handle regular questions with safety check
                reply = "The question is considered as a violation!!!!!"
                if safety_check(prompt):
                    reply = qna(prompt, file_id)["answer"]
                else:
                    logger.warning("Prompt violation detected.")

            db.log_chat(file_id, reply, role="assistant")
            return {"response": reply}
        except Exception as e:
            logger.error(f"Chatbot command execution failed: {str(e)}")
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )


def db_smr(file_id: str):
    """
    Returns the summary for a given PDF file.

    Args:
        file_id (str): ID of the file to summarize.

    Returns:
        JSONResponse: Summary content or error.
    """
    with Database() as db:
        try:
            logger.info(f"Retrieving summary for file: {file_id}")
            text = db.get_smr_by_id(file_id)[0]
            return JSONResponse(content={"summary": text})
        except Exception as e:
            logger.error(f"Summary retrieval failed: {str(e)}")
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )


def db_remove_pdf():
    """
    Clears temporary PDF files from the system.

    Returns:
        JSONResponse: Success or no-operation status.
    """
    if clear_pdf():
        logger.info("Temporary files cleared successfully.")
        return JSONResponse(content={"success": "File cleared"})
    logger.info("No temporary file to clear.")
    return JSONResponse(
        content={"status": "No file to be cleared"},
        status_code=200,
    )
