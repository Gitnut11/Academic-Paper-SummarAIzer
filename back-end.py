from fastapi import FastAPI, UploadFile, File, Form
from database import Database
from fastapi.responses import JSONResponse, FileResponse
import os

app = FastAPI()


@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    db = Database()
    success = db.create_user(username, password)
    db.close()
    if success:
        return {"message": "User registered successfully"}
    return JSONResponse(content={"error": "Username already exists"}, status_code=400)


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    db = Database()
    user_id = db.login(username, password)
    if user_id:
        file_list = db.get_user_files(user_id)
        db.close()
        return {"user_id": user_id, "files": file_list}
    db.close()
    return JSONResponse(content={"error": "Invalid username or password"}, status_code=401)


@app.post("/upload")
def upload_file(user_id: str = Form(...), file: UploadFile = File(...)):
    db = Database()
    try:
        file_binary = file.file.read()
        db.insert_user_file_binary(user_id, file_binary, file.filename)
        file_list = db.get_user_files(user_id)
        db.close()
        return file_list
    except Exception as e:
        db.close()
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/pdf/{id}")
def pdf(id: int):
    db = Database()
    try:
        file_path, file_name = db.get_file_by_id(id)
        db.close()
        if os.path.exists(file_path):
            return FileResponse(path=file_path, filename=file_name + ".pdf", media_type="application/pdf")
        return JSONResponse(content={"error": "File not found"}, status_code=500)
    except Exception as e:
        db.close()
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/chat/{file_id}")
def chat(file_id: int):
    db = Database()
    try:
        chat_history = db.get_history(file_id)
        db.close()
        return JSONResponse(content={"history": chat_history})
    except Exception as e:
        db.close()
        return JSONResponse(content={"error": str(e)}, status_code=500)



@app.post("/chatbot")
def chatbot(prompt: str = Form(...), file_id: int = Form(...)):
    db = Database()
    try:
        reply = "" # DO STH <----------------------------------- process model here
        db.log_chat(file_id, prompt, role="user")
        db.log_chat(file_id, reply, role="assistant")
        return {"response": reply}
    except Exception as e:
        db.close()
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
