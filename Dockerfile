FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy source and precomputed artifacts
COPY src ./src
COPY main.py ./main.py
COPY artifacts ./artifacts
COPY data/processed ./data/processed

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi[all]>=0.118.0 \
    uvicorn[standard]>=0.37.0 \
    pydantic>=2.11.0 \
    numpy>=2.0.0 \
    pandas>=3.0.1 \
    pyarrow>=23.0.1 \
    scipy>=1.17.1 \
    faiss-cpu>=1.11.0 \
    rapidfuzz>=3.12.0 \
    httpx>=0.28.0 \
    sentence-transformers>=2.2.2

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]