#!/usr/bin/env python3
"""
CS627 Semantic-Vector Space - Command Line Interface

Advanced workflows and pipeline orchestration for the CS627 project.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
import click
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    click.echo(f"{BLUE}Running: {cmd}{RESET}")
    return subprocess.run(cmd, shell=True, check=check, capture_output=False, text=True)


def check_tokens_exist() -> bool:
    """Check if pre-generated tokens exist."""
    token_path = Path("data/pregenerated_tokens/mosi/train_tokens.h5")
    return token_path.exists()


def check_docker() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run("docker --version", shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """CS627 Semantic-Vector Space CLI - Advanced workflow automation."""
    pass


# ============================================================================
# Pipeline Commands
# ============================================================================

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Pipeline configuration file')
@click.option('--skip-tokens', is_flag=True, help='Skip token generation if exists')
@click.option('--skip-training', is_flag=True, help='Skip training step')
@click.option('--skip-evaluation', is_flag=True, help='Skip evaluation step')
@click.option('--device', type=click.Choice(['cuda', 'cpu', 'auto']), default='auto',
              help='Device for training')
def pipeline(config, skip_tokens, skip_training, skip_evaluation, device):
    """Run complete training pipeline from data to evaluation."""
    
    click.echo(f"{BOLD}{GREEN}CS627 Training Pipeline{RESET}")
    click.echo("=" * 50)
    
    # Load configuration if provided
    if config:
        with open(config, 'r') as f:
            cfg = json.load(f)
        click.echo(f"Loaded configuration from: {config}")
    else:
        cfg = {}
    
    # Step 1: Check prerequisites
    click.echo(f"\n{BOLD}Step 1: Checking prerequisites...{RESET}")
    if not check_docker():
        click.echo(f"{RED}✗ Docker not found! Please install Docker first.{RESET}")
        sys.exit(1)
    click.echo(f"{GREEN}✓ Docker is available{RESET}")
    
    # Step 2: Token generation
    if not skip_tokens:
        click.echo(f"\n{BOLD}Step 2: Generating tokens...{RESET}")
        if check_tokens_exist():
            click.echo(f"{YELLOW}Tokens already exist. Use --skip-tokens to skip.{RESET}")
            if click.confirm("Regenerate tokens?"):
                run_command("make tokens")
        else:
            run_command("make tokens")
    else:
        click.echo(f"\n{BOLD}Step 2: Skipping token generation{RESET}")
    
    # Step 3: Training
    if not skip_training:
        click.echo(f"\n{BOLD}Step 3: Training adapters...{RESET}")
        
        # Set device
        os.environ['DEVICE'] = device
        
        # Get training parameters from config
        batch_size = cfg.get('batch_size', 256)
        epochs = cfg.get('epochs', 50)
        lr = cfg.get('learning_rate', 0.001)
        
        cmd = f"BATCH_SIZE={batch_size} EPOCHS={epochs} LEARNING_RATE={lr} "
        if device == 'cpu':
            cmd += "make train-cpu"
        else:
            cmd += "make train"
        
        run_command(cmd)
    else:
        click.echo(f"\n{BOLD}Step 3: Skipping training{RESET}")
    
    # Step 4: Evaluation
    if not skip_evaluation:
        click.echo(f"\n{BOLD}Step 4: Running evaluation...{RESET}")
        run_command("make evaluate")
    else:
        click.echo(f"\n{BOLD}Step 4: Skipping evaluation{RESET}")
    
    click.echo(f"\n{GREEN}{BOLD}Pipeline complete!{RESET}")


# ============================================================================
# Experiment Commands
# ============================================================================

@cli.command()
@click.option('--configs', type=click.Choice(['recommended', 'grid_search', 'random_search',
                                               'progressive_depth', 'activation_study', 
                                               'dropout_study']),
              default='recommended', help='Configuration set to run')
@click.option('--config-file', type=click.Path(exists=True), help='Custom config file')
@click.option('--epochs', type=int, default=30, help='Epochs per configuration')
@click.option('--parallel', type=int, default=1, help='Number of parallel experiments')
def ablation(configs, config_file, epochs, parallel):
    """Run ablation studies to find optimal adapter configurations."""
    
    click.echo(f"{BOLD}{GREEN}Ablation Study Runner{RESET}")
    click.echo("=" * 50)
    
    if not check_tokens_exist():
        click.echo(f"{RED}✗ No tokens found! Please run 'make tokens' first.{RESET}")
        sys.exit(1)
    
    if config_file:
        cmd = f"python src/Experiments/run_ablation.py --config-file {config_file}"
    else:
        cmd = f"python src/Experiments/run_ablation.py --config {configs}"
    
    cmd += f" --epochs {epochs}"
    
    if parallel > 1:
        click.echo(f"Running {parallel} experiments in parallel...")
        # TODO: Implement parallel execution
        click.echo(f"{YELLOW}Parallel execution not yet implemented, running sequentially{RESET}")
    
    run_command(cmd)
    
    # Show results
    results_dir = Path("Results/ablation")
    if results_dir.exists():
        latest = max(results_dir.glob("*.json"), key=os.path.getmtime, default=None)
        if latest:
            click.echo(f"\n{BOLD}Results saved to: {latest}{RESET}")
            
            # Parse and show best config
            with open(latest, 'r') as f:
                results = json.load(f)
            
            if isinstance(results, list) and len(results) > 0:
                best = min(results, key=lambda x: x.get('best_val_loss', float('inf')))
                click.echo(f"\n{BOLD}Best Configuration:{RESET}")
                click.echo(f"  Hidden Size: {best['config']['hidden_size']}")
                click.echo(f"  Layers: {best['config']['num_layers']}")
                click.echo(f"  Activation: {best['config']['activation']}")
                click.echo(f"  Val Loss: {best['best_val_loss']:.4f}")


# ============================================================================
# Benchmark Commands
# ============================================================================

@cli.command()
@click.option('--dataset', type=click.Choice(['mosi', 'mosei', 'meld', 'iemocap']),
              default='mosi', help='Dataset to benchmark on')
@click.option('--metrics', multiple=True, default=['recall', 'similarity'],
              help='Metrics to compute')
@click.option('--output', type=click.Path(), help='Output file for results')
def benchmark(dataset, metrics, output):
    """Run benchmarks on trained models."""
    
    click.echo(f"{BOLD}{GREEN}Running Benchmarks{RESET}")
    click.echo("=" * 50)
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Metrics: {', '.join(metrics)}")
    
    # Check if models exist
    weights_dir = Path("OptimalWeights")
    if not weights_dir.exists() or not list(weights_dir.glob("*.pth")):
        click.echo(f"{RED}✗ No trained models found in OptimalWeights/{RESET}")
        sys.exit(1)
    
    # Run evaluation
    cmd = "python src/Evaluation/run_benchmarks.py"
    cmd += f" --dataset {dataset}"
    for metric in metrics:
        cmd += f" --metric {metric}"
    
    if output:
        cmd += f" --output {output}"
    
    run_command(cmd, check=False)
    
    # Display results
    if output and Path(output).exists():
        with open(output, 'r') as f:
            results = json.load(f)
        
        click.echo(f"\n{BOLD}Benchmark Results:{RESET}")
        for metric, value in results.items():
            if isinstance(value, float):
                click.echo(f"  {metric}: {value:.3f}")
            else:
                click.echo(f"  {metric}: {value}")


# ============================================================================
# Data Commands
# ============================================================================

@cli.command()
@click.option('--dataset', type=click.Choice(['mosi', 'mosei', 'all']), default='mosi',
              help='Dataset to download')
@click.option('--extract', is_flag=True, help='Extract audio/video after download')
def data(dataset, extract):
    """Download and prepare datasets."""
    
    click.echo(f"{BOLD}{GREEN}Data Preparation{RESET}")
    click.echo("=" * 50)
    
    if dataset in ['mosi', 'all']:
        click.echo("Downloading CMU-MOSI dataset...")
        run_command("make download-data")
    
    if dataset in ['mosei', 'all']:
        click.echo("CMU-MOSEI download not yet implemented")
        # TODO: Implement MOSEI download
    
    if extract:
        click.echo("Extracting audio and video segments...")
        run_command("make extract-data")
    
    click.echo(f"{GREEN}✓ Data preparation complete!{RESET}")


# ============================================================================
# Model Commands
# ============================================================================

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--text', type=str, help='Text input to encode')
@click.option('--audio', type=click.Path(exists=True), help='Audio file to encode')
@click.option('--image', type=click.Path(exists=True), help='Image file to encode')
@click.option('--output', type=click.Path(), help='Output file for embeddings')
def encode(model_path, text, audio, image, output):
    """Encode inputs using trained models."""
    
    click.echo(f"{BOLD}{GREEN}Encoding Inputs{RESET}")
    click.echo("=" * 50)
    
    # Import here to avoid loading models unnecessarily
    from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
    import torch
    import numpy as np
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = {}
    
    if text:
        click.echo(f"Encoding text: '{text[:50]}...'")
        encoder = Text_to_Vec().to(device)
        encoder.eval()
        with torch.no_grad():
            embedding = encoder([text]).cpu().numpy()
        embeddings['text'] = embedding.tolist()
    
    if audio:
        click.echo(f"Encoding audio: {audio}")
        encoder = Audio_to_Vec().to(device)
        encoder.eval()
        # Load and encode audio
        # TODO: Implement audio loading
        click.echo(f"{YELLOW}Audio encoding not yet implemented{RESET}")
    
    if image:
        click.echo(f"Encoding image: {image}")
        encoder = Image_to_Vec().to(device)
        encoder.eval()
        from PIL import Image
        img = Image.open(image)
        with torch.no_grad():
            embedding = encoder([img]).cpu().numpy()
        embeddings['image'] = embedding.tolist()
    
    # Save or display embeddings
    if output:
        with open(output, 'w') as f:
            json.dump(embeddings, f, indent=2)
        click.echo(f"Embeddings saved to: {output}")
    else:
        click.echo("\nEmbeddings:")
        for modality, emb in embeddings.items():
            click.echo(f"  {modality}: shape {np.array(emb).shape}")


# ============================================================================
# Utility Commands
# ============================================================================

@cli.command()
def status():
    """Show current training status and results."""
    
    click.echo(f"{BOLD}{GREEN}Training Status{RESET}")
    click.echo("=" * 50)
    
    # Check tokens
    if check_tokens_exist():
        token_path = Path("data/pregenerated_tokens/mosi")
        sizes = []
        for f in token_path.glob("*.h5"):
            size_mb = f.stat().st_size / (1024 * 1024)
            sizes.append(f"{f.name}: {size_mb:.1f}MB")
        click.echo(f"{GREEN}✓ Tokens generated{RESET}")
        for s in sizes:
            click.echo(f"  {s}")
    else:
        click.echo(f"{RED}✗ No tokens found{RESET}")
    
    # Check weights
    weights_dir = Path("OptimalWeights")
    if weights_dir.exists():
        weights = list(weights_dir.glob("*.pth"))
        if weights:
            click.echo(f"\n{GREEN}✓ {len(weights)} trained models found{RESET}")
            for w in weights[:5]:
                size_mb = w.stat().st_size / (1024 * 1024)
                click.echo(f"  {w.name}: {size_mb:.1f}MB")
        else:
            click.echo(f"\n{RED}✗ No trained models found{RESET}")
    
    # Check results
    results_dir = Path("Results/adapter_training")
    if results_dir.exists():
        results = list(results_dir.glob("*.json"))
        if results:
            click.echo(f"\n{GREEN}✓ {len(results)} training results found{RESET}")
            latest = max(results, key=os.path.getmtime)
            click.echo(f"  Latest: {latest.name}")
            
            # Show latest metrics
            with open(latest, 'r') as f:
                data = json.load(f)
            
            if 'history' in data:
                final_loss = data['history'].get('val_loss', [])[-1] if data['history'].get('val_loss') else None
                if final_loss:
                    click.echo(f"  Final val loss: {final_loss:.4f}")


@cli.command()
@click.option('--deep', is_flag=True, help='Deep clean including Docker volumes')
@click.option('--yes', is_flag=True, help='Skip confirmation')
def clean(deep, yes):
    """Clean generated files and caches."""
    
    click.echo(f"{BOLD}{YELLOW}Cleanup{RESET}")
    click.echo("=" * 50)
    
    items = [
        "data/pregenerated_tokens/",
        "Results/",
        "__pycache__",
        ".pytest_cache"
    ]
    
    if deep:
        items.extend([
            "OptimalWeights/*.pth",
            "CandidateWeights/",
            "training_reports/"
        ])
        click.echo(f"{YELLOW}Deep clean will remove trained models!{RESET}")
    
    click.echo("Will remove:")
    for item in items:
        click.echo(f"  - {item}")
    
    if not yes:
        if not click.confirm("Continue?"):
            click.echo("Cancelled")
            return
    
    run_command("make clean")
    
    if deep:
        click.echo("Running deep clean...")
        run_command("rm -rf OptimalWeights/*.pth CandidateWeights/ training_reports/")
    
    click.echo(f"{GREEN}✓ Cleanup complete!{RESET}")


@cli.command()
@click.option('--port', default=8888, help='Jupyter port')
def dev(port):
    """Start development environment with Jupyter."""
    
    click.echo(f"{BOLD}{GREEN}Starting Development Environment{RESET}")
    click.echo("=" * 50)
    click.echo(f"Jupyter will be available at: http://localhost:{port}")
    click.echo("Press Ctrl+C to stop")
    
    cmd = f"JUPYTER_PORT={port} docker-compose up dev"
    run_command(cmd, check=False)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    cli()