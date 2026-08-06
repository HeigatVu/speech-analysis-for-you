# Speech Analysis for You (SAY)

## 🎯 Overview
This pipeline processes speech audio recordings and extracts clinically relevant features that can distinguish between healthy speech patterns and those affected by cognitive decline. The system is designed with modularity and extensibility in mind, allowing for easy integration of machine learning models in future iterations.

### Vietnamese speech feature library (research-only)

The `src/speech_features` package is a **math-first, research-only** feature
library for AD-versus-healthy-control speech studies. It processes participant
WAV recordings plus reviewed, aligned Vietnamese transcripts for picture
description, immediate/delayed recall, and phonemic/semantic fluency tasks. Key
points you should know before using it:

- **Not a diagnostic and not for clinical decision-making.** These features and
  the research baseline are for cohort characterisation only. They do not give a
  diagnosis and define **no fixed score threshold**. We do not promise any fixed
  performance; do not treat any metric as a guarantee of real-world accuracy.
- **Math-first extraction.** Feature extraction uses NumPy/SciPy and stdlib
  primitives only — no librosa, openSMILE, spaCy, ASR, embeddings, or
  transformer models.
- **Deterministic contracts.** Standard PCM WAV in (down-mixed to mono and resampled to 16 kHz), reviewed Vietnamese transcript JSON (a transcript, not whitespace, defines word boundaries) in,
  immutable `FeatureResult` with per-input SHA-256 provenance out. See
  [Feature Extraction](docs/feature-extraction.md) for the full
  formulas, manifest/task-spec fields, quality flags, label separation, batch
  failure behaviour, and the repeated nested grouped-CV research baseline.
- **Tests first.** Every module was built test-first; the docs are guarded by
  `tests/speech_features/test_documentation.py`.

### Key features:
- 🔧 Robust Preprocessing: 
- 🎵 Multi-domain Feature Extraction:
- 📊 Statistical Analysis: 
- 📈 Visualization Tools:

## Table of Contents
- [Speech Analysis for You (SAY)](#speech-analysis-for-you-say)
  - [🎯 Overview](#-overview)
    - [Key features:](#key-features)
    - [Vietnamese speech feature library (research-only)](#vietnamese-speech-feature-library-research-only)
  - [Table of Contents](#table-of-contents)
  - [🚀 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Install UV](#install-uv)
    - [Clone and Setup](#clone-and-setup)
  - [⚡ Quick Start](#-quick-start)
    - [1. Prepare Your Data](#1-prepare-your-data)
    - [2. Configure the Pipeline](#2-configure-the-pipeline)
    - [3. Run the Pipeline](#3-run-the-pipeline)
    - [4. View Results](#4-view-results)
  - [📁 Project Structure](#-project-structure)
  - [📂 Output Directory Structure](#-output-directory-structure)
    - [Feature File Format (JSON)](#feature-file-format-json)
    - [Combined Features File](#combined-features-file)
  - [📖 Usage](#-usage)
    - [Preprocessing Audio](#preprocessing-audio)
    - [Feature Extraction](#feature-extraction)
    - [Feature Analysis](#feature-analysis)
  - [⚙️ Configuration](#️-configuration)
    - [Feature Extraction Configuration](#feature-extraction-configuration)
    - [Preprocessing Configuration](#preprocessing-configuration)
  - [📋 Tasks](#-tasks)
  - [📊 Feature Documentation](#-feature-documentation)
    - [Acoustic Features](#acoustic-features)
    - [Linguistic Features](#linguistic-features)
  - [📝 Citation](#-citation)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)
  - [📧 Contact](#-contact)
  - [🗺️ Roadmap](#️-roadmap)

## 🚀 Installation
### Prerequisites

- Python 3.10 or higher
- [UV package manager](https://github.com/astral-sh/uv)

### Install UV

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/HeigatVu/speech-analysis-for-you.git
cd speech-analysis-for-you

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate # On linux/macOs
.venv\Scripts\activate # On windows

# Install the package
uv pip install -e ".[dev]"
```

## ⚡ Quick Start
### 1. Prepare Your Data
Place your audio files in the following structure:
```
data/raw/audio/
├── participant_001/
│   ├── recording_01.wav
│   └── recording_02.wav
└── participant_002/
    └── recording_01.wav
    └── recording_02.wav
```
### 2. Configure the Pipeline


### 3. Run the Pipeline


### 4. View Results


## 📁 Project Structure
```
speech-analysis-for-you/
├── config/                      # Configuration files
│   ├── feature_extraction.yaml  # Feature extraction parameters
│   ├── preprocessing.yaml       # Audio preprocessing settings
│   └── paths.yaml               # Path configurations
├── src/speech_features/         # Main package
│   ├── io/                      # Input/Output operations
│   ├── preprocessing/           # Audio preprocessing
│   ├── feature_extraction/      # Feature extraction modules
│   │   ├── acoustic/            # Acoustic features
│   │   └── linguistic/          # Linguistic features
│   ├── analysis/                # Feature analysis tools
│   └── utils/                   # Utility functions
├── scripts/                     # Executable scripts
└── notebooks/                   # Jupyter notebooks for analyzing
```

## 📂 Output Directory Structure
```
speech-analysis-for-you/
│
├── data/
│   ├── raw/audio/                    # 📥 INPUT: Original audio files
│   │   ├── participant_001/
│   │   │   └── recording_01.wav
│   │   └── participant_002/
│   │       └── recording_01.wav
│   │
│   ├── processed/                    # 🔧 INTERMEDIATE: Preprocessed audio
│   │   ├── participant_001/
│   │   │   └── recording_01.wav      # Cleaned, normalized, silence removed
│   │   └── participant_002/
│   │       └── recording_01.wav
│   │
│   └── features/                     # ✨ OUTPUT: Extracted features
│       ├── participant_001/
│       │   └── recording_01.json     # All features for this recording
│       ├── participant_002/
│       │   └── recording_01.json
│       └── all_features.json         # 📊 Combined: all recordings
│
└── output/                          # 📈 Analysis results
    ├── report/
    └── figure/
```

### Feature File Format (JSON)

### Combined Features File


## 📖 Usage
### Preprocessing Audio

### Feature Extraction

### Feature Analysis

## ⚙️ Configuration
Configuration files are located in the `config/` directory and use YAML format.
### Feature Extraction Configuration

### Preprocessing Configuration

## 📋 Tasks
Q1: Maximum Phonation Time

Q2: Read sentences

Q3: Immediately recall the story

Q4: Picture description

Q5: Picture story narrative

Q6: Phonetic fluency task

Q7: Senmantic fluency task

Q8: Procedural discourse task

Q9: Simple calculation task

Q10: Calling object task

Q11: Delayed story recall task

## 📊 Feature Documentation
### Acoustic Features

### Linguistic Features

## 📝 Citation
If you use this work in your research, please cite:
```bibtex
@software{speech_analysis_for_you_202x,
  title = {SAY: a speech analysis for You},
  author = {Vu Nguyen Minh Duc},
  year = {202x},
  url = {https://github.com/HeigatVu/speech-analysis-for-you.git}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SAY dataset contributors
- Internaltional University - Vietnam National University Speech Processing resources

## 📧 Contact

**Vu Nguyen Minh Duc** - [vnmduc.work@gmail.com](mailto:vnmduc.work@gmail.com)

Project Link: [https://github.com/HeigatVu/speech-analysis-for-you.git](https://github.com/HeigatVu/speech-analysis-for-you.git)

## 🗺️ Roadmap

- [x] Basic feature extraction pipeline
- [x] Comprehensive preprocessing
- [x] Feature analysis tools
- [ ] Integration with ML models (planned)
- [ ] Real-time feature extraction (in the future)
- [ ] Web-based demo interface (in the future)

---