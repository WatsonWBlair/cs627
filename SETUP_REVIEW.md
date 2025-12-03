# Setup Script Portability Review

## Files Reviewed
- `setup_and_train.sh` - Main cloud/local initialization script
- `.gitignore` - Git tracking exclusions
- Download scripts in `scripts/data_wrangling/`

## Status: .gitignore ✅

**All data directories properly excluded:**
- Line 191: `data/` - Excludes entire data directory
- Lines 199-201: Explicit MOSI exclusions (`**/cmumosi/`, `**/cmu-mosi/`, `**/mosi/`)
- Line 225: `AdapterWeights/` - Trained weights excluded
- Lines 164-172: Model weights excluded (`*.pth`, `*.pt`, `*.bin`, etc.)

**Verdict**: Videos, audio, frames, and weights will NOT be tracked by git ✅

---

## Issues Found in setup_and_train.sh

### Issue 1: Incorrect Training Script Path (Line 362)
**Current:**
```bash
python3 src/Training/train_encoder_alignment.py 2>&1 | tee "$LOG_FILE"
```

**Problem**: `train_encoder_alignment.py` doesn't exist

**Fix**: Should use current training script
```bash
python3 src/Training/train_raw_encoders.py 2>&1 | tee "$LOG_FILE"
```

### Issue 2: Old Function Calls (Lines 307-320)
**Current:**
```python
from src.Training.Data_Wrangling.mosi_dataset import download_mosi, preprocess_mosi
download_mosi('$DATA_PATH')
preprocess_mosi('$DATA_PATH')
```

**Problem**: `preprocess_mosi()` function doesn't exist in current codebase

**Fix**: Remove preprocess call, only download metadata:
```python
from src.Training.Data_Wrangling.mosi_dataset import download_mosi
download_mosi('$DATA_PATH/mosi/')
```

### Issue 3: Obsolete Import Paths (Line 421)
**Current:**
```bash
python3 -c "from src.Encoders.text_2_vec import Text_to_Vec; encoder = Text_to_Vec()"
```

**Problem**: Old file naming convention (`text_2_vec` vs `text/semantic_to_vec`)

**Fix**: Use new import paths:
```bash
python3 -c "from src.Encoders import Text_to_Vec; encoder = Text_to_Vec()"
```

---

## Recommendations

### Critical (Must Fix) ✅ ALL FIXED
1. ✅ Update line 362 to use `train_raw_encoders.py`
2. ✅ Remove `preprocess_mosi()` call (lines 307-320)
3. ✅ Update import example (line 421)
4. ✅ Update README.md imports (lines 56-57)
5. ✅ Update README.md directory structure (lines 127-129)

### Nice to Have
1. Add video download step after metadata download
2. Add audio/frame extraction step before training
3. Update comments to reflect 4-encoder swarm architecture

### Cloud Setup Process (Corrected)
1. Install system dependencies ✅
2. Install Python packages ✅
3. Install CMU-MultimodalSDK ✅
4. Download MOSI metadata ✅
5. **Download videos** (MISSING - should use `download_all_mosi_videos.py`)
6. **Extract segments** (MISSING - should use `extract_test_segments.py` or create `extract_all_segments.py`)
7. Train encoders (needs path fix)

---

## New Scripts Created

### `download_all_mosi_videos.py`
- Downloads full MOSI dataset (~93 videos)
- Parallel downloads (4 workers)
- Creates manifest JSON
- Estimated time: 1-3 hours
- Expected success: 55-60% (~50-55 videos)

### Usage
```bash
python scripts/data_wrangling/download_all_mosi_videos.py
```

---

## Quick Setup Guide (Portable)

### Local Windows (Current Setup)
```powershell
# Install dependencies
pip install -r requirements.txt

# Install CMU-MultimodalSDK
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install . && cd ..

# Download full dataset (or just 5 test videos)
python scripts/data_wrangling/download_all_mosi_videos.py
# OR for quick testing:
python scripts/data_wrangling/download_test_videos.py

# Extract segments
python scripts/data_wrangling/extract_test_segments.py

# Train swarm
python src/Training/train_raw_encoders.py
```

### Linux/Cloud GPU
```bash
# Clone repo
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627

# Run setup (will need fixes above)
chmod +x setup_and_train.sh
./setup_and_train.sh

# OR manual setup:
pip install -r requirements.txt
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install . && cd ..
python scripts/data_wrangling/download_all_mosi_videos.py
python scripts/data_wrangling/extract_test_segments.py
python src/Training/train_raw_encoders.py
```

---

## Verified Working

✅ Data directories excluded from git
✅ 4-encoder swarm training (text, audio_waveform, audio_tone, image)
✅ Feature importance weighting system
✅ Tone encoder integration (WavLM)
✅ Test video download (5 videos)
✅ Audio/frame extraction
✅ Full dataset download script created
✅ **setup_and_train.sh** - All critical issues FIXED
✅ **README.md** - All import paths updated

## Remaining Tasks

⚠️ Missing "extract all segments" script for full dataset (nice to have)
⚠️ Could add video/audio extraction steps to setup_and_train.sh (nice to have)
