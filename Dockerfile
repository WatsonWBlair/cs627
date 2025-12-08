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
             /workspace/data/pregenerated_tokens/mosi \
             /workspace/OptimalWeights \
             /workspace/CandidateWeights \
             /workspace/Results/adapter_training \
             /workspace/Results/ablation \
             /workspace/configs/ablation \
             /workspace/training_reports

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Environment variables for training configuration
ENV DEVICE=cuda \
    BATCH_SIZE=32 \
    LEARNING_RATE=0.0001 \
    EPOCHS=30 \
    SEMANTIC_DIM=1024 \
    ADAPTER_HIDDEN_SIZE=512 \
    ADAPTER_LAYERS=2 \
    USE_AMP=1 \
    GRADIENT_ACCUMULATION=1

# Data paths
ENV MOSI_DATA_PATH=/workspace/data/cmumosi/mosi/ \
    AUDIO_DIR=/workspace/data/cmumosi/audio/ \
    VIDEO_DIR=/workspace/data/cmumosi/frames/ \
    OUTPUT_DIR=/workspace/data/pregenerated_tokens \
    TOKEN_DIR=/workspace/data/pregenerated_tokens/mosi \
    OPTIMAL_WEIGHTS_DIR=/workspace/OptimalWeights \
    CANDIDATE_WEIGHTS_DIR=/workspace/CandidateWeights \
    RESULTS_DIR=/workspace/Results/adapter_training

# S3 configuration (set USE_S3=1 to enable S3 data loading)
ENV USE_S3=0 \
    S3_BUCKET=cs627-svs-data \
    S3_REGION=us-east-1 \
    S3_DATA_PREFIX=data/cmumosi/ \
    S3_WEIGHTS_PREFIX=OptimalWeights/

# Health check - verify GPU is accessible
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')" || exit 1

# Default command - show GPU info and usage instructions
CMD ["bash", "-c", "echo '=== CS627 Semantic-Vector Space Training Environment ===' && \
     echo 'MIT Licensed for Educational and Research Use' && \
     echo '' && \
     python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"Device count: {torch.cuda.device_count()}\") if torch.cuda.is_available() else print(\"Running in CPU mode\")' && \
     echo '' && \
     echo 'Token-Based Training Workflow:' && \
     echo '  Step 1: python src/Training/pregenerate_tokens.py' && \
     echo '  Step 2: python src/Training/train_adapters.py --mode encoder' && \
     echo '  Ablation: python src/Experiments/run_ablation.py' && \
     echo '' && \
     echo 'Mount points:' && \
     echo '  /workspace/data - Training data and tokens' && \
     echo '  /workspace/OptimalWeights - Model weights' && \
     echo '  /workspace/Results - Training results' && \
     echo '' && \
     /bin/bash"]