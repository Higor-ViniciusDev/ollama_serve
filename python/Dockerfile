# Base Python
FROM python:3.11-slim

# Evita interações durante instalação
ENV DEBIAN_FRONTEND=noninteractive

# Atualiza pacotes e instala dependências básicas
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential curl git nano && \
    rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Copia requirements
COPY requirements.txt .

# Instala bibliotecas Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia código
COPY . .

# Comando padrão
CMD ["bash"]
