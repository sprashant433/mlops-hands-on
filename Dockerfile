FROM python:3.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY src ./src
RUN mkdir -p ./mlruns
# COPY mlruns ./mlruns

EXPOSE 8000

CMD ["python", "src/mlops_lr/serve.py"]
