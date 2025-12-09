@echo off
REM CS627 Semantic-Vector Space - Windows Make Equivalent
REM Provides similar functionality to Makefile for Windows users

setlocal enabledelayedexpansion

if "%1"=="" goto help
goto %1

:help
echo CS627 Semantic-Vector Space - Available Commands
echo ================================================
echo.
echo Setup ^& Installation:
echo   make setup          Install dependencies and prepare environment
echo   make docker-build   Build Docker images locally
echo.
echo Data Preparation:
echo   make download-data  Download CMU-MOSI dataset
echo   make extract-data   Extract audio/video from downloaded data
echo   make prepare-data   Run full data preparation pipeline
echo.
echo Training Pipeline:
echo   make tokens         Generate encoder tokens (GPU Docker)
echo   make tokens-cpu     Generate encoder tokens (CPU Docker)
echo   make tokens-local   Generate encoder tokens (no Docker)
echo   make train          Train adapters (GPU Docker)
echo   make train-cpu      Train adapters (CPU Docker)
echo   make train-local    Train adapters (no Docker)
echo   make ablation       Run ablation study (Docker)
echo   make ablation-local Run ablation study (no Docker)
echo   make visualize-ablation  Visualize ablation study results
echo   make visualize-performance  Visualize training performance
echo   make visualize-decoder  Visualize decoder training results
echo.
echo Evaluation ^& Testing:
echo   make evaluate       Run evaluation on trained models
echo   make test           Run unit tests
echo   make smoke-test     Run quick smoke test
echo.
echo Development:
echo   make dev            Start Jupyter development environment
echo   make lint           Run code linting
echo   make format         Auto-format code
echo.
echo Utilities:
echo   make clean          Remove generated files and caches
echo   make status         Show training status and results
echo   make logs           Show recent training logs
echo.
echo Docker Hub:
echo   make docker-push    Push images to Docker Hub
echo   make docker-pull    Pull latest images from Docker Hub
goto end

:setup
echo Setting up CS627 environment...
pip install -r requirements.txt
if not exist "CMU-MultimodalSDK" (
    echo Installing CMU-MultimodalSDK...
    git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
    cd CMU-MultimodalSDK
    pip install .
    cd ..
)
if not exist "data\cmumosi\mosi" mkdir data\cmumosi\mosi
if not exist "data\cmumosi\audio" mkdir data\cmumosi\audio
if not exist "data\cmumosi\frames" mkdir data\cmumosi\frames
if not exist "data\cmumosi\videos" mkdir data\cmumosi\videos
if not exist "data\pregenerated_tokens\mosi" mkdir data\pregenerated_tokens\mosi
if not exist "OptimalWeights" mkdir OptimalWeights
if not exist "Results\adapter_training" mkdir Results\adapter_training
if not exist "Results\ablation" mkdir Results\ablation
if not exist "configs\ablation" mkdir configs\ablation
echo Setup complete!
goto end

:docker-build
echo Building Docker images...
docker build -f Dockerfile -t watsonwb/cs627-svs:gpu -t watsonwb/cs627-svs:latest .
if exist "Dockerfile.cpu" (
    docker build -f Dockerfile.cpu -t watsonwb/cs627-svs:cpu .
)
echo Docker images built successfully!
goto end

:download-data
echo Downloading CMU-MOSI dataset...
python -c "from src.Training.Data_Wrangling.mosi_dataset import download_mosi; download_mosi('data/cmumosi/mosi/')"
echo Data download complete!
goto end

:extract-data
echo Extracting audio and video segments...
if exist "scripts\data_wrangling\extract_test_segments.py" (
    python scripts\data_wrangling\extract_test_segments.py
) else (
    echo Extract script not found, skipping...
)
echo Data extraction complete!
goto end

:prepare-data
call :download-data
call :extract-data
echo Data preparation complete!
goto end

:tokens
echo Generating encoder tokens...
docker-compose up pregenerate-tokens
echo Token generation complete!
goto end

:tokens-cpu
echo Generating encoder tokens (CPU)...
docker-compose up pregenerate-tokens-cpu
echo Token generation complete!
goto end

:tokens-local
echo Generating encoder tokens locally (no Docker)...
python src/Training/pregenerate_tokens.py
echo Token generation complete!
goto end

:train
echo Training adapters...
if not exist "data\pregenerated_tokens\mosi\train_tokens.h5" (
    echo No tokens found! Running token generation first...
    call :tokens
)
docker-compose up train-adapters-gpu
echo Adapter training complete!
goto end

:train-cpu
echo Training adapters on CPU...
if not exist "data\pregenerated_tokens\mosi\train_tokens.h5" (
    echo No tokens found! Running token generation first...
    call :tokens
)
docker-compose up train-adapters-cpu
echo CPU training complete!
goto end

:train-local
echo Training adapters locally (no Docker)...
if not exist "data\pregenerated_tokens\mosi\train_tokens.h5" (
    echo No tokens found! Run 'make.bat tokens-local' first.
    goto end
)
python src/Training/train_adapters.py --mode encoder
echo Training complete!
goto end

:ablation
echo Running ablation study...
if not exist "data\pregenerated_tokens\mosi\train_tokens.h5" (
    echo No tokens found! Running token generation first...
    call :tokens
)
docker-compose up ablation-study
echo Ablation study complete!
goto end

:ablation-local
echo Running ablation study locally (no Docker)...
if not exist "data\pregenerated_tokens\mosi\train_tokens.h5" (
    echo No tokens found! Run 'make.bat tokens-local' first.
    goto end
)
python src/Experiments/run_ablation.py --config recommended
echo Ablation study complete!
goto end

:visualize-ablation
echo Visualizing ablation study results...
if not exist "Results\ablation" (
    echo No ablation results found! Run 'make.bat ablation-local' first.
    goto end
)
python scripts/visualize_ablation.py
echo Visualization complete! Check Results/ablation/figures/
goto end

:visualize-performance
echo Generating performance figures...
if not exist "OptimalWeights" (
    echo No trained weights found! Run 'make.bat train-local' first.
    goto end
)
python scripts/generate_performance_figures.py
echo Visualization complete! Check figures/
goto end

:visualize-decoder
echo Generating decoder training figures...
if not exist "DecoderCheckpoints" (
    echo No decoder checkpoints found! Train decoders first.
    goto end
)
python scripts/generate_decoder_figures.py
echo Visualization complete! Check figures/decoders/
goto end

:evaluate
echo Running evaluation...
docker-compose up evaluate
echo Evaluation complete!
goto end

:test
echo Running unit tests...
python -m pytest tests\ -v
echo Tests complete!
goto end

:smoke-test
echo Running smoke test...
docker-compose up test
echo Smoke test complete!
goto end

:dev
echo Starting Jupyter development environment...
echo Access at: http://localhost:8888
docker-compose up dev
goto end

:lint
echo Running code linting...
python -m flake8 src\ --max-line-length=100 --ignore=E203,W503
python -m mypy src\ --ignore-missing-imports
echo Linting complete!
goto end

:format
echo Auto-formatting code...
python -m black src\ tests\ scripts\
python -m isort src\ tests\ scripts\
echo Formatting complete!
goto end

:clean
echo Cleaning generated files...
if exist "data\pregenerated_tokens" rmdir /s /q data\pregenerated_tokens
if exist "Results" rmdir /s /q Results
if exist "__pycache__" rmdir /s /q __pycache__
if exist ".pytest_cache" rmdir /s /q .pytest_cache
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
del /s /q *.pyc 2>nul
echo Cleanup complete!
goto end

:status
echo Training Status:
echo ================
if exist "data\pregenerated_tokens\mosi\train_tokens.h5" (
    echo Tokens generated
    dir data\pregenerated_tokens\mosi\*.h5 2>nul
) else (
    echo No tokens found
)
echo.
if exist "OptimalWeights\*.pth" (
    echo Adapter weights found:
    dir OptimalWeights\*.pth 2>nul | findstr /i ".pth"
) else (
    echo No trained adapters found
)
echo.
if exist "Results\adapter_training\*.json" (
    echo Training results:
    dir Results\adapter_training\*.json 2>nul | findstr /i ".json"
) else (
    echo No training results found
)
goto end

:logs
echo Recent training logs:
if exist "Results\adapter_training" (
    dir Results\adapter_training\*.json /o-d 2>nul | findstr /i ".json" | head -5
) else (
    echo No logs found
)
goto end

:docker-push
echo Pushing images to Docker Hub...
docker push watsonwb/cs627-svs:gpu
docker push watsonwb/cs627-svs:latest
docker images | findstr "watsonwb/cs627-svs.*cpu" >nul 2>&1
if %errorlevel%==0 (
    docker push watsonwb/cs627-svs:cpu
)
echo Images pushed successfully!
goto end

:docker-pull
echo Pulling latest images from Docker Hub...
docker pull watsonwb/cs627-svs:latest
docker pull watsonwb/cs627-svs:gpu
docker pull watsonwb/cs627-svs:cpu
echo Images pulled successfully!
goto end

:quick-train
call :tokens
call :train
goto end

:pipeline
call :prepare-data
call :tokens
call :train
call :evaluate
goto end

:reset
call :clean
call :setup
goto end

:check
call :format
call :lint
call :test
goto end

:end
endlocal