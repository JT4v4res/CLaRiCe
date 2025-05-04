FROM python:3.11-bookworm

# Atualiza pip
RUN pip install --upgrade pip

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia requirements e instala dependências
COPY requirements.txt .

RUN pip install --no-cache-dir huggingface_hub
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='JT4v4res/CLaRiCe')"

RUN pip install --no-cache-dir -r requirements.txt

RUN rm -rf requirements.txt

# Baixa o modelo do spaCy
RUN python -m spacy download pt_core_news_lg

# Copia o conteúdo da pasta src/ para dentro de /app
COPY ./src/ .

EXPOSE 8000

# Garante que o diretório atual está no PYTHONPATH
ENV PYTHONPATH="/app"

# Executa a aplicação
CMD ["python", "main.py"]

