# Use the latest official CUDA runtime image from NVIDIA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive 

# Install Python and other dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

# Install Python packages globally
RUN pip install model-compression-toolkit pandas onnx torch torchvision

# Create a group with GID 200
RUN groupadd -g 200 appuser

# Create a user with UID 9895 and associate it with GID 200
RUN useradd -u 9895 -g 200 -m -s /bin/bash appuser

# Change ownership of the /app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser
