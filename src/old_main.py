import logging  # For logging events and errors

from fastapi import FastAPI  # FastAPI framework for building APIs
from fastapi.middleware.cors import (
    CORSMiddleware,
)  # Middleware for handling Cross-Origin Resource Sharing

from qna import PDFRequest, QnaRequest, read_pdf, clear_pdf, qna
from summarize import SmrRequest, summarize

if not logging.getLogger().handlers:
    with open(".log", "w") as f:
        f.write("")
    # Create a log file if it doesn't exist
    logging.basicConfig(
        filename=".log",  # The log file path
        level=logging.INFO,  # Log level; change to DEBUG for more detailed output
        format="%(asctime)s - %(levelname)s - %(message)s",  # Format for each log message
    )

app = FastAPI()
logging.info("FastAPI server started in main.py")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development; restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def on_shutdown():
    clear_pdf()

@app.post("/smr")
async def get_smr(request: SmrRequest):
    return summarize(request)

@app.post("/qna")
async def get_qna(request: QnaRequest):
    return qna(request)

@app.post("/pdf")
async def load_pdf(request: PDFRequest):
    return read_pdf(request)

@app.get("/pdf-clear")
async def remove_pdf():
    return clear_pdf()
\
#======== Comment when using docker ===========================================
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
#==============================================================================
    