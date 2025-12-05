# CS627 Semantic-Vector Space Training Docker Image (GPU-enabled)
# MIT License - For educational and research use
# Author: Watson Blair

FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Metadata
LABEL maintainer="Watson Blair"
LABEL description="CS627 Semantic-Vector Space multimodal training environment with GPU support"
LABEL license="MIT"
LABEL version="1.0.0"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libsndfile1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install CMU-MultimodalSDK
RUN git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git /tmp/CMU-MultimodalSDK && \
    cd /tmp/CMU-MultimodalSDK && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/CMU-MultimodalSDK

# Copy source code
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/
COPY tests/ /workspace/tests/
COPY tools/ /workspace/tools/

# Copy documentation and license
COPY LICENSE /workspace/
COPY CLAUDE.md /workspace/
COPY README.md /workspace/
COPY *.md /workspace/

# Create necessary directories
RUN mkdir -p /workspace/data/cmumosi/{mosi,audio,frames,videos} \
             /workspace/OptimalWeights \
             /workspace/CandidateWeights \
             /workspace/DecoderCheckpoints \
             /workspace/training_reports

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Environment variables for training configuration
ENV DEVICE=cuda \
    BATCH_SIZE=32 \
    LEARNING_RATE=0.0001 \
    EPOCHS=30 \
    MOMENTUM=0.999 \
    QUEUE_SIZE=4096 \
    TEMPERATURE=0.07 \
    SEMANTIC_DIM=1024

# Data paths
ENV MOSI_DATA_PATH=/workspace/data/cmumosi/mosi/ \
    AUDIO_DIR=/workspace/data/cmumosi/audio/ \
    VIDEO_DIR=/workspace/data/cmumosi/frames/ \
    OPTIMAL_WEIGHTS_DIR=/workspace/OptimalWeights \
    CANDIDATE_WEIGHTS_DIR=/workspace/CandidateWeights

# Health check - verify GPU is accessible
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')" || exit 1

# Default command - show GPU info and usage instructions
CMD ["bash", "-c", "echo '=== CS627 Semantic-Vector Space Training Environment ===' && \
     echo 'MIT Licensed for Educational and Research Use' && \
     echo '' && \
     python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"Device count: {torch.cuda.device_count()}\") if torch.cuda.is_available() else print(\"Running in CPU mode\")' && \
     echo '' && \
     echo 'Usage:' && \
     echo '  Training: python src/Training/train_encoders.py' && \
     echo '  Testing: python -m pytest tests/' && \
     echo '' && \
     echo 'Mount points:' && \
     echo '  /workspace/data - Training data' && \
     echo '  /workspace/OptimalWeights - Model weights' && \
     echo '' && \
     /bin/bash"]