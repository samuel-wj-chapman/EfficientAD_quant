FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install dependencies
RUN apt-get update
RUN apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    libgl1-mesa-glx \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.8
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python packages
RUN pip install /app

RUN pip install model-compression-toolkit onnx


