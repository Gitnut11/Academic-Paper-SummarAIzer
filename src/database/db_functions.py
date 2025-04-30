import os

from fastapi import UploadFile
from fastapi.responses import FileResponse, JSONResponse

from utils.prompt import HELPER
from database.database import Database
from models.qna import PDFProcessor, clear_pdf, current_pdf, qna, read_pdf
from models.summarize import summarize
from utils.pdf_utils import get_sections


def db_register(username: str, password: str) -> dict | JSONResponse:
    with Database() as db:
        if db.create_user(username, password):
            return {"message": "User registered successfully"}
        return JSONResponse(
            content={"error": "Username already exists"},
            status_code=400,
        )


def db_login(username: str, password: str) -> dict | JSONResponse:
    with Database() as db:
        user_id = db.login(username, password)
        if user_id:
            file_list = db.get_user_files(user_id)
            return {
                "user_id": user_id,
                "files": file_list,
            }
        return JSONResponse(
            content={"error": "Invalid username or password"},
            status_code=401,
        )


def db_upload_file(user_id: str, file: UploadFile):
    with Database() as db:
        try:
            file_binary = file.file.read()
            db.insert_user_file_binary(user_id, file_binary, file.filename)
            file_list = db.get_user_files(user_id)
            return file_list
        except Exception as e:
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )


def db_get_pdf(file_id: int):
    with Database() as db:
        try:
            file_path, file_name = db.get_file_by_id(file_id)
            if os.path.exists(file_path):
                return FileResponse(
                    path=file_path,
                    filename=f"{file_name}.pdf",
                    media_type="application/pdf",
                )
            return JSONResponse(
                content={"error": "File not found"},
                status_code=400,
            )
        except Exception as e:
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )


def db_get_chat(file_id: int):
    with Database() as db:
        try:
            chat_hist = db.get_history(file_id)
            return JSONResponse(content={"history": chat_hist})
        except Exception as e:
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )


def db_chatbot(prompt: str, file_id: int):
    with Database() as db:
        try:
            db.log_chat(file_id, prompt, role="user")
            prompt = prompt.strip()
            command = prompt.split()[0]

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
                reply = """"""
                for section in sections:
                    reply += f"""{section}\n\n"""
            elif command[0] == "/":
                reply = """
                    Unknown command detected, use `/help` for some instruction
                """
            else:
                reply = qna(prompt)["answer"]

            db.log_chat(file_id, reply, roloe="assistant")
            return {"response": reply}
        except Exception as e:
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )


def db_smr(file_id: int):
    with Database() as db:
        try:
            text = db.get_smr_by_id(file_id)[0]
            return JSONResponse(content={"summary": text})
        except Exception as e:
            return JSONResponse(
                content={"error": str(e)},
                status_code=500,
            )


def db_load_pdf(file_id: int):
    with Database() as db:
        last = current_pdf()
        file = db.get_file_by_id(file_id)

        if file is None:
            return JSONResponse(
                content={"error": "File not found"},
                status_code=400,
            )
        file_path = file[0]
        result = read_pdf(file_path)
        if last != result:
            return JSONResponse(
                content={"success": "Chose a new pdf"},
                status_code=400,
            )
        return JSONResponse(
            content={"error": "Internal error"},
            status_code=500,
        )


def db_remove_pdf():
    if clear_pdf():
        return JSONResponse(content={"success": "File cleared"})
    return JSONResponse(
        content={"status": "No file to be cleared"},
        status_code=200,
    )
