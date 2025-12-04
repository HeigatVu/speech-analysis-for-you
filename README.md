# Speech-Analysis-for-You

## 🎯 Features
- Audio Processing:
- Feature Extraction:

## 📋 Requirements
- Python 3.12
- UV package manager
- CUDA-capable GPU (recommended)

## 🚀 Quick Start
### 1. Install UV:
```
https://github.com/astral-sh/uv
```  
### 2. Setup Project:
```
# Navigate to project directory
cd 
```


```
alzheimer-speech-detection/
├── assets/              # Static resources (pre-trained models, references)
├── config/              # Configuration files (YAML)
├── data/               # Dataset (raw, processed, external)
├── src/                # Source code
│   ├── data/          # Data processing & PyTorch datasets
│   ├── features/      # Feature extraction
│   └── utils/         # Utilities (config, logging, audio)
├── scripts/           # Executable scripts
├── notebooks/         # Jupyter notebooks for analysis
└── results/           # Experiment results
```

# Flow to run
Raw data -> preprocessing (clapperboard-detection.ipynb) in processed_data with two dir segments and slate_positions -> Rename file into participant_001_Qx_sub.wav ..... 

# Note
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