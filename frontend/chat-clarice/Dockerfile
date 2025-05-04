FROM python:3.11-bookworm

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN rm -rf requirements.txt

RUN rm -rf ./src/DEPRECATED.py

COPY ./src/ .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]