# Use the official slim Python 3.11 image as a base for a smaller footprint
FROM python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the necessary modules
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache

RUN python -m spacy download en_core_web_sm && \
    python -m nltk.downloader punkt punkt_tab && \
    rm -rf /root/.cache

# Copy the entire source code into the container's working directory
COPY . .

# Expose port 8000 so it can be accessed from the host
EXPOSE 8000

# Start the FastAPI application using uvicorn when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
