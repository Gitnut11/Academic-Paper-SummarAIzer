from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import os
from database import Database
from qna import qna, PDFProcessor
from summarize import summarize
from pdf_utils import get_sections
from pdf_utils import extract_page_as_binary

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Allow all origins (for development; restrict in production)
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    db = Database()
    success = db.create_user(username, password)
    db.close()
    if success:
        return {"message": "User registered successfully"}
    return JSONResponse(content={"error": "Username already exists"}, status_code=400)


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    db = Database()
    user_id = db.login(username, password)
    if user_id:
        file_list = db.get_user_files(user_id)
        db.close()
        return {"user_id": user_id, "files": file_list}
    db.close()
    return JSONResponse(content={"error": "Invalid username or password"}, status_code=401)


@app.post("/upload")
async def upload_file(user_id: str = Form(...), file: UploadFile = File(...)):
    db = Database()
    try:
        file_binary = file.file.read()
        db.insert_user_file_binary(user_id, file_binary, file.filename)
        file_list = db.get_user_files(user_id)
        print(file_list)
        db.close()
        return file_list
    except Exception as e:
        db.close()
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/pdf/{id}/{num}")
async def pdf(id: str, num: int):
    db = Database()
    try:
        file_path, file_name = db.get_file_by_id(id)
        if not os.path.exists(file_path):
            return JSONResponse(content={"error": "File not found"}, status_code=400)
        binary = extract_page_as_binary(file_path, num)
        db.close()
        return Response(content=binary, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename={file_name}_{num}.pdf"})
    except Exception as e:
        db.close()
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/chat/{file_id}")
async def chat(file_id: str):
    db = Database()
    try:
        chat_history = db.get_history(file_id)
        db.close()
        return JSONResponse(content={"history": chat_history})
    except Exception as e:
        db.close()
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/chatbot")
async def chatbot(prompt: str = Form(...), file_id: str = Form(...)):
    db = Database()
    try:
        db.log_chat(file_id, prompt, role="user")
        prompt = prompt.strip()
        command = prompt.split()[0]
        if command == '/help':
            reply = '''
                `/summarize`: to summarize \n
                `/sections`: show list of detected sections \n
                Other inputs are treated as Q&A prompt for this paper!
            '''
        elif command == '/summarize':
            file = db.get_file_by_id(file_id)
            pdf_process = PDFProcessor()
            text = pdf_process.extract_text(file[0])
            reply = summarize(text)
        elif command == '/sections':
            file = db.get_file_by_id(file_id)
            sections = get_sections(file[0])
            reply = ''''''
            for section in sections:
                reply += f'''{section}\n\n'''
        elif command[0] == '/':
            reply = '''
                Unknown command detected, use `/help` for some instruction
            '''
        else:
            reply = qna(prompt, file_id)["answer"]
        db.log_chat(file_id, reply, role="assistant")
        return {"response": reply}
    except Exception as e:
        db.close()
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/smr/{file_id}")
async def smr(file_id: str):
    db = Database()
    try:
        text = db.get_smr_by_id(file_id)[0]
        db.close()
        return JSONResponse(content={"summary": text})
    except Exception as e:
        db.close()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# @app.get("/pdf-clear")
# async def remove_pdf():
#     if clear_pdf():
#         return JSONResponse(content={"success": "file cleared"})
#     return JSONResponse(content={"status": "No file to be cleared"}, status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
