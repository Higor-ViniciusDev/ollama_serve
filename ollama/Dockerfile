FROM ollama/ollama:latest

USER root

RUN apt-get update && \
    apt-get install -y apt-utils sudo curl wget nano vim git gnupg && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility