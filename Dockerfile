FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "gradio==4.44.1"

COPY . .

RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

CMD ["python", "app.py"]
