FROM python:3.11-bookworm

RUN pip install --upgrade pip

RUN pip install --no-cache-dir requests

WORKDIR /app

COPY . .

CMD ["python", "model_fetch.py"]