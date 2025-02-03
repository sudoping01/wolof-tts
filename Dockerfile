
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV PYTHON_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    ffmpeg \
    gfortran \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/Galsenaicommunity/Wolof-TTS.git

WORKDIR /app/Wolof-TTS/notebooks/Models/xTTS\ v2

RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir spacy imbalanced-learn plotly ipython

RUN pip3 install --no-cache-dir --upgrade "numpy<2.0"

COPY galsenai-xtts-wo-checkpoints.zip .

RUN unzip galsenai-xtts-wo-checkpoints.zip && rm galsenai-xtts-wo-checkpoints.zip


COPY app.py .

EXPOSE 8080

CMD ["python3", "app.py"]
