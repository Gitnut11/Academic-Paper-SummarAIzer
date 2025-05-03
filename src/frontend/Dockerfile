FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend .

ARG API_URL=http://backend:8000
ENV API_URL=${API_URL}

# 4. Patch the URL in your Streamlit script at container start, then run
EXPOSE 8501
CMD ["bash","-c","sed -i \"s|http://localhost:8000|${API_URL}|g\" main.py && streamlit run main.py --server.address 0.0.0.0 --server.port 8501"]