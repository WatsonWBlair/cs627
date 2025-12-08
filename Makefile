# CS627 Semantic-Vector Space - Makefile
# Simplifies common tasks for training and evaluation

.PHONY: help setup tokens train evaluate clean docker-build docker-push test lint format

# Default target - show help
help:
	@echo "CS627 Semantic-Vector Space - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          Install dependencies and prepare environment"
	@echo "  make docker-build   Build Docker images locally"
	@echo ""
	@echo "Data Preparation:"
	@echo "  make download-data  Download CMU-MOSI dataset"
	@echo "  make extract-data   Extract audio/video from downloaded data"
	@echo "  make prepare-data   Run full data preparation pipeline"
	@echo ""
	@echo "Training Pipeline:"
	@echo "  make tokens         Generate encoder tokens (run once)"
	@echo "  make train          Train adapters with pre-generated tokens"
	@echo "  make train-cpu      Train adapters on CPU"
	@echo "  make ablation       Run ablation study"
	@echo ""
	@echo "Evaluation & Testing:"
	@echo "  make evaluate       Run evaluation on trained models"
	@echo "  make test           Run unit tests"
	@echo "  make smoke-test     Run quick smoke test"
	@echo ""
	@echo "Development:"
	@echo "  make dev            Start Jupyter development environment"
	@echo "  make lint           Run code linting"
	@echo "  make format         Auto-format code"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          Remove generated files and caches"
	@echo "  make status         Show training status and results"
	@echo "  make logs           Show recent training logs"
	@echo ""
	@echo "Docker Hub:"
	@echo "  make docker-push    Push images to Docker Hub"
	@echo "  make docker-pull    Pull latest images from Docker Hub"

# ============================================================================
# Setup & Installation
# ============================================================================

setup:
	@echo "Setting up CS627 environment..."
	@pip install -r requirements.txt
	@if [ ! -d "CMU-MultimodalSDK" ]; then \
		echo "Installing CMU-MultimodalSDK..."; \
		git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git; \
		cd CMU-MultimodalSDK && pip install .; \
	fi
	@mkdir -p data/cmumosi/{mosi,audio,frames,videos}
	@mkdir -p data/pregenerated_tokens/mosi
	@mkdir -p OptimalWeights Results/adapter_training Results/ablation
	@mkdir -p configs/ablation
	@echo "✓ Setup complete!"

docker-build:
	@echo "Building Docker images..."
	@docker build -f Dockerfile -t watsonwb/cs627-svs:gpu -t watsonwb/cs627-svs:latest .
	@if [ -f "Dockerfile.cpu" ]; then \
		docker build -f Dockerfile.cpu -t watsonwb/cs627-svs:cpu .; \
	fi
	@echo "✓ Docker images built successfully!"

# ============================================================================
# Data Preparation
# ============================================================================

download-data:
	@echo "Downloading CMU-MOSI dataset..."
	@python -c "from src.Training.Data_Wrangling.mosi_dataset import download_mosi; download_mosi('data/cmumosi/mosi/')"
	@echo "✓ Data download complete!"

extract-data:
	@echo "Extracting audio and video segments..."
	@if [ -f "scripts/data_wrangling/extract_test_segments.py" ]; then \
		python scripts/data_wrangling/extract_test_segments.py; \
	else \
		echo "Extract script not found, skipping..."; \
	fi
	@echo "✓ Data extraction complete!"

prepare-data: download-data extract-data
	@echo "✓ Data preparation complete!"

# ============================================================================
# Training Pipeline
# ============================================================================

tokens:
	@echo "Generating encoder tokens..."
	@if [ -f "docker/pregenerate.sh" ]; then \
		./docker/pregenerate.sh; \
	else \
		docker-compose up pregenerate-tokens; \
	fi
	@echo "✓ Token generation complete!"

train:
	@echo "Training adapters..."
	@if [ ! -f "data/pregenerated_tokens/mosi/train_tokens.h5" ]; then \
		echo "No tokens found! Running token generation first..."; \
		$(MAKE) tokens; \
	fi
	@if [ -f "docker/train-adapters.sh" ]; then \
		./docker/train-adapters.sh --mode encoder; \
	else \
		docker-compose up train-adapters-gpu; \
	fi
	@echo "✓ Adapter training complete!"

train-cpu:
	@echo "Training adapters on CPU..."
	@if [ ! -f "data/pregenerated_tokens/mosi/train_tokens.h5" ]; then \
		echo "No tokens found! Running token generation first..."; \
		$(MAKE) tokens; \
	fi
	@docker-compose up train-adapters-cpu
	@echo "✓ CPU training complete!"

ablation:
	@echo "Running ablation study..."
	@if [ ! -f "data/pregenerated_tokens/mosi/train_tokens.h5" ]; then \
		echo "No tokens found! Running token generation first..."; \
		$(MAKE) tokens; \
	fi
	@docker-compose up ablation-study
	@echo "✓ Ablation study complete!"

# ============================================================================
# Evaluation & Testing
# ============================================================================

evaluate:
	@echo "Running evaluation..."
	@docker-compose up evaluate
	@echo "✓ Evaluation complete!"

test:
	@echo "Running unit tests..."
	@python -m pytest tests/ -v
	@echo "✓ Tests complete!"

smoke-test:
	@echo "Running smoke test..."
	@docker-compose up test
	@echo "✓ Smoke test complete!"

# ============================================================================
# Development
# ============================================================================

dev:
	@echo "Starting Jupyter development environment..."
	@echo "Access at: http://localhost:8888"
	@docker-compose up dev

lint:
	@echo "Running code linting..."
	@python -m flake8 src/ --max-line-length=100 --ignore=E203,W503
	@python -m mypy src/ --ignore-missing-imports
	@echo "✓ Linting complete!"

format:
	@echo "Auto-formatting code..."
	@python -m black src/ tests/ scripts/
	@python -m isort src/ tests/ scripts/
	@echo "✓ Formatting complete!"

# ============================================================================
# Utilities
# ============================================================================

clean:
	@echo "Cleaning generated files..."
	@rm -rf data/pregenerated_tokens/
	@rm -rf Results/
	@rm -rf __pycache__ .pytest_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "✓ Cleanup complete!"

status:
	@echo "Training Status:"
	@echo "================"
	@if [ -f "data/pregenerated_tokens/mosi/train_tokens.h5" ]; then \
		echo "✓ Tokens generated"; \
		ls -lh data/pregenerated_tokens/mosi/*.h5 2>/dev/null | tail -3; \
	else \
		echo "✗ No tokens found"; \
	fi
	@echo ""
	@if [ -d "OptimalWeights" ] && [ "$$(ls -A OptimalWeights)" ]; then \
		echo "✓ Adapter weights found:"; \
		ls -lt OptimalWeights/*.pth 2>/dev/null | head -5; \
	else \
		echo "✗ No trained adapters found"; \
	fi
	@echo ""
	@if [ -d "Results/adapter_training" ] && [ "$$(ls -A Results/adapter_training)" ]; then \
		echo "✓ Training results:"; \
		ls -lt Results/adapter_training/*.json 2>/dev/null | head -3; \
	else \
		echo "✗ No training results found"; \
	fi

logs:
	@echo "Recent training logs:"
	@if [ -d "Results/adapter_training" ]; then \
		find Results/adapter_training -name "*.json" -type f -exec ls -lt {} + 2>/dev/null | head -5; \
	else \
		echo "No logs found"; \
	fi

# ============================================================================
# Docker Hub Operations
# ============================================================================

docker-push:
	@echo "Pushing images to Docker Hub..."
	@docker push watsonwb/cs627-svs:gpu
	@docker push watsonwb/cs627-svs:latest
	@if docker images | grep -q "watsonwb/cs627-svs.*cpu"; then \
		docker push watsonwb/cs627-svs:cpu; \
	fi
	@echo "✓ Images pushed successfully!"

docker-pull:
	@echo "Pulling latest images from Docker Hub..."
	@docker pull watsonwb/cs627-svs:latest
	@docker pull watsonwb/cs627-svs:gpu
	@docker pull watsonwb/cs627-svs:cpu
	@echo "✓ Images pulled successfully!"

# ============================================================================
# Quick Commands (Aliases)
# ============================================================================

# Quick train - generates tokens if needed, then trains
quick-train: tokens train

# Full pipeline - data prep, tokens, training, evaluation
pipeline: prepare-data tokens train evaluate

# Reset and start fresh
reset: clean setup

# Development workflow - format, lint, test
check: format lint test