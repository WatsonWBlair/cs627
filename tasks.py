"""
CS627 Semantic-Vector Space - Cross-Platform Task Runner

Usage:
    inv --list           # Show all available tasks
    inv setup            # Install dependencies
    inv train            # Train adapters
    inv tokens           # Generate tokens

Replaces both Makefile and make.bat with a single Python-based solution.
Requires: pip install invoke
"""

import os
import sys
import shutil
from pathlib import Path
from invoke import task, Collection

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TOKENS_DIR = DATA_DIR / "pregenerated_tokens" / "mosi"
WEIGHTS_DIR = PROJECT_ROOT / "OptimalWeights"
RESULTS_DIR = PROJECT_ROOT / "Results"

DOCKER_IMAGE = "watsonwb/cs627-svs"


# =============================================================================
# Helper Functions
# =============================================================================

def has_tokens():
    """Check if pre-generated tokens exist."""
    return (TOKENS_DIR / "train_tokens.h5").exists()


def has_weights():
    """Check if trained adapter weights exist."""
    return WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.glob("*.pth"))


def print_header(msg):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


# =============================================================================
# Setup & Installation
# =============================================================================

@task
def setup(c):
    """Install dependencies and prepare environment."""
    print_header("Setting up CS627 environment")

    # Install Python dependencies
    c.run("pip install -r requirements.txt")

    # Install CMU-MultimodalSDK if not present
    if not (PROJECT_ROOT / "CMU-MultimodalSDK").exists():
        print("Installing CMU-MultimodalSDK...")
        c.run("git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git")
        with c.cd("CMU-MultimodalSDK"):
            c.run("pip install .")

    # Create directory structure
    dirs = [
        "data/cmumosi/mosi",
        "data/cmumosi/audio",
        "data/cmumosi/frames",
        "data/cmumosi/videos",
        "data/pregenerated_tokens/mosi",
        "OptimalWeights",
        "Results/adapter_training",
        "Results/ablation",
        "configs/ablation",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("\n[OK] Setup complete!")


@task
def docker_build(c):
    """Build Docker images locally."""
    print_header("Building Docker images")

    c.run(f"docker build -f Dockerfile -t {DOCKER_IMAGE}:gpu -t {DOCKER_IMAGE}:latest .")

    if (PROJECT_ROOT / "Dockerfile.cpu").exists():
        c.run(f"docker build -f Dockerfile.cpu -t {DOCKER_IMAGE}:cpu .")

    print("\n[OK] Docker images built successfully!")


# =============================================================================
# Data Preparation
# =============================================================================

@task
def download_data(c):
    """Download CMU-MOSI dataset."""
    print_header("Downloading CMU-MOSI dataset")

    c.run('python -c "from src.Training.Data_Wrangling.mosi_dataset import download_mosi; download_mosi(\'data/cmumosi/mosi/\')"')

    print("\n[OK] Data download complete!")


@task
def extract_data(c):
    """Extract audio and video segments from downloaded data."""
    print_header("Extracting audio and video segments")

    script = PROJECT_ROOT / "scripts" / "data_wrangling" / "extract_test_segments.py"
    if script.exists():
        c.run(f"python {script}")
    else:
        print("Extract script not found, skipping...")

    print("\n[OK] Data extraction complete!")


@task(pre=[download_data, extract_data])
def prepare_data(c):
    """Run full data preparation pipeline."""
    print("\n[OK] Data preparation complete!")


# =============================================================================
# Training Pipeline
# =============================================================================

@task
def tokens(c, local=False, cpu=False):
    """
    Generate encoder tokens.

    Args:
        local: Run locally without Docker
        cpu: Use CPU-only Docker image
    """
    print_header("Generating encoder tokens")

    if local:
        c.run("python src/Training/pregenerate_tokens.py")
    elif cpu:
        c.run("docker-compose up pregenerate-tokens-cpu")
    else:
        # Try shell script first, fall back to docker-compose
        if (PROJECT_ROOT / "docker" / "pregenerate.sh").exists():
            c.run("./docker/pregenerate.sh")
        else:
            c.run("docker-compose up pregenerate-tokens")

    print("\n[OK] Token generation complete!")


@task
def train(c, mode="encoder", local=False, cpu=False):
    """
    Train adapters with pre-generated tokens.

    Args:
        mode: Training mode - 'encoder' or 'decoder'
        local: Run locally without Docker
        cpu: Use CPU-only Docker image
    """
    print_header(f"Training {mode} adapters")

    # Check for tokens
    if not has_tokens():
        print("No tokens found! Running token generation first...")
        tokens(c, local=local, cpu=cpu)

    if local:
        c.run(f"python src/Training/train_adapters.py --mode {mode}")
    elif cpu:
        c.run("docker-compose up train-adapters-cpu")
    else:
        if (PROJECT_ROOT / "docker" / "train-adapters.sh").exists():
            c.run(f"./docker/train-adapters.sh --mode {mode}")
        else:
            c.run("docker-compose up train-adapters-gpu")

    print(f"\n[OK] {mode.capitalize()} adapter training complete!")


@task
def ablation(c, local=False, config="recommended"):
    """
    Run ablation study.

    Args:
        local: Run locally without Docker
        config: Ablation config name (default: recommended)
    """
    print_header("Running ablation study")

    if not has_tokens():
        print("No tokens found! Run 'inv tokens' first.")
        return

    if local:
        c.run(f"python src/Experiments/run_ablation.py --config {config}")
    else:
        c.run("docker-compose up ablation-study")

    print("\n[OK] Ablation study complete!")


# =============================================================================
# Visualization
# =============================================================================

@task
def visualize_ablation(c):
    """Visualize ablation study results."""
    print_header("Visualizing ablation study results")

    if not (RESULTS_DIR / "ablation").exists():
        print("No ablation results found! Run 'inv ablation --local' first.")
        return

    c.run("python scripts/visualize_ablation.py")
    print("\n[OK] Visualization complete! Check Results/ablation/figures/")


@task
def visualize_performance(c):
    """Generate performance figures from training results."""
    print_header("Generating performance figures")

    if not has_weights():
        print("No trained weights found! Run 'inv train --local' first.")
        return

    c.run("python scripts/generate_performance_figures.py")
    print("\n[OK] Visualization complete! Check figures/")


@task
def visualize_decoder(c):
    """Generate decoder training visualization figures."""
    print_header("Generating decoder training figures")

    if not (PROJECT_ROOT / "DecoderCheckpoints").exists():
        print("No decoder checkpoints found! Train decoders first.")
        return

    c.run("python scripts/generate_decoder_figures.py")
    print("\n[OK] Visualization complete! Check figures/decoders/")


# =============================================================================
# Evaluation & Testing
# =============================================================================

@task
def evaluate(c, local=False):
    """Run evaluation on trained models."""
    print_header("Running evaluation")

    if local:
        c.run("python src/Evaluation/run_evaluation.py")
    else:
        c.run("docker-compose up evaluate")

    print("\n[OK] Evaluation complete!")


@task
def test(c, verbose=True):
    """Run unit tests."""
    print_header("Running unit tests")

    flags = "-v" if verbose else ""
    c.run(f"python -m pytest tests/ {flags}")

    print("\n[OK] Tests complete!")


@task
def smoke_test(c):
    """Run quick smoke test."""
    print_header("Running smoke test")

    c.run("docker-compose up test")

    print("\n[OK] Smoke test complete!")


# =============================================================================
# Development
# =============================================================================

@task
def dev(c):
    """Start Jupyter development environment."""
    print_header("Starting Jupyter development environment")
    print("Access at: http://localhost:8888")

    c.run("docker-compose up dev")


@task
def lint(c):
    """Run code linting."""
    print_header("Running code linting")

    c.run("python -m flake8 src/ --max-line-length=100 --ignore=E203,W503", warn=True)
    c.run("python -m mypy src/ --ignore-missing-imports", warn=True)

    print("\n[OK] Linting complete!")


@task
def format(c):
    """Auto-format code with black and isort."""
    print_header("Auto-formatting code")

    c.run("python -m black src/ tests/ scripts/")
    c.run("python -m isort src/ tests/ scripts/")

    print("\n[OK] Formatting complete!")


# =============================================================================
# Utilities
# =============================================================================

@task
def clean(c):
    """Remove generated files and caches."""
    print_header("Cleaning generated files")

    # Remove directories
    dirs_to_remove = [
        "data/pregenerated_tokens",
        "Results",
        "__pycache__",
        ".pytest_cache",
    ]
    for d in dirs_to_remove:
        path = PROJECT_ROOT / d
        if path.exists():
            shutil.rmtree(path)
            print(f"  Removed {d}")

    # Remove __pycache__ directories recursively
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache)

    # Remove .pyc files
    for pyc in PROJECT_ROOT.rglob("*.pyc"):
        pyc.unlink()

    print("\n[OK] Cleanup complete!")


@task
def status(c):
    """Show training status and results."""
    print_header("Training Status")

    # Check tokens
    if has_tokens():
        print("[OK] Tokens generated")
        for f in TOKENS_DIR.glob("*.h5"):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"     {f.name}: {size_mb:.1f} MB")
    else:
        print("[  ] No tokens found")

    print()

    # Check weights
    if has_weights():
        print("[OK] Adapter weights found:")
        for f in sorted(WEIGHTS_DIR.glob("*.pth"))[:5]:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"     {f.name}: {size_mb:.1f} MB")
    else:
        print("[  ] No trained adapters found")

    print()

    # Check results
    results_dir = RESULTS_DIR / "adapter_training"
    if results_dir.exists() and any(results_dir.glob("*.json")):
        print("[OK] Training results:")
        for f in sorted(results_dir.glob("*.json"))[:3]:
            print(f"     {f.name}")
    else:
        print("[  ] No training results found")


@task
def logs(c):
    """Show recent training logs."""
    print_header("Recent training logs")

    results_dir = RESULTS_DIR / "adapter_training"
    if results_dir.exists():
        files = sorted(results_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        for f in files[:5]:
            print(f"  {f.name}")
    else:
        print("No logs found")


# =============================================================================
# Docker Hub Operations
# =============================================================================

@task
def docker_push(c):
    """Push images to Docker Hub."""
    print_header("Pushing images to Docker Hub")

    c.run(f"docker push {DOCKER_IMAGE}:gpu")
    c.run(f"docker push {DOCKER_IMAGE}:latest")

    # Check if CPU image exists
    result = c.run(f"docker images | grep '{DOCKER_IMAGE}.*cpu'", warn=True, hide=True)
    if result.ok:
        c.run(f"docker push {DOCKER_IMAGE}:cpu")

    print("\n[OK] Images pushed successfully!")


@task
def docker_pull(c):
    """Pull latest images from Docker Hub."""
    print_header("Pulling images from Docker Hub")

    c.run(f"docker pull {DOCKER_IMAGE}:latest")
    c.run(f"docker pull {DOCKER_IMAGE}:gpu")
    c.run(f"docker pull {DOCKER_IMAGE}:cpu", warn=True)

    print("\n[OK] Images pulled successfully!")


# =============================================================================
# Composite Tasks (Pipelines)
# =============================================================================

@task
def quick_train(c, local=False):
    """Generate tokens if needed, then train adapters."""
    tokens(c, local=local)
    train(c, local=local)


@task
def pipeline(c, local=False):
    """Full pipeline: data prep, tokens, training, evaluation."""
    prepare_data(c)
    tokens(c, local=local)
    train(c, local=local)
    evaluate(c, local=local)


@task(pre=[clean, setup])
def reset(c):
    """Reset environment: clean and setup fresh."""
    print("\n[OK] Environment reset complete!")


@task
def check(c):
    """Development workflow: format, lint, test."""
    format(c)
    lint(c)
    test(c)
