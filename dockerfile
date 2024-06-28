# Use the latest official CUDA runtime image from NVIDIA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive 
# Install Python and other dependencies
RUN apt-get update --fix-missing
RUN apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python packages
# Ensure you have a requirements.txt file or install directly
RUN pip install /app

# Alternatively, if specific packages are needed, install them directly
RUN pip install model-compression-toolkit onnx

