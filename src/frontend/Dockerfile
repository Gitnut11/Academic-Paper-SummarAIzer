FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Upgrade pip and install Python dependencies without cache to keep image small
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache

COPY . .

# Define a build-time argument for the backend API URL (can be overridden with `--build-arg`)
ARG API_URL=http://backend:8000

# Set the environment variable that will be used at runtime
ENV API_URL=${API_URL}

# Expose the default Streamlit port
EXPOSE 8501

# At runtime, replace all hardcoded backend URLs in main.py with the provided API_URL,
# then start the Streamlit server on all interfaces (0.0.0.0) at port 8501
CMD ["bash","-c","sed -i \"s|http://localhost:8000|${API_URL}|g\" main.py && streamlit run main.py --server.address 0.0.0.0 --server.port 8501"]
