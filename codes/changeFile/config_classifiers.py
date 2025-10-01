## Config for acoustic analysis
# Cross-validation parameters
N_FOLDS = 5
CV_SCORER = 'AUC'  # Options: 'AUC', 'KAPPA', 'accuracy'

# Classification parameters
DEFAULT_CLASSIFIERS = [1, 3]  # LR and SVM
CLASS_TYPES = ['2-way']  # ['2-way', '3-way']

# Feature extraction parameters
DEFAULT_FEATURE_SETS = ['eGeMAPSv02', 'ComParE_2016']

# Audio processing parameters
DEFAULT_SAMPLE_RATE = None  # Let librosa auto-detect
SEGMENT_LENGTH = 30  # seconds (for future segmentation if needed)

# File naming conventions
AUDIO_FILE_PATTERNS = {
    'picture_description': '*_picture_description.wav',
    'short_memory': '*_short_memory.wav', 
    'long_memory': '*_long_memory.wav',
    'semantic_fluency': '*_semantic_fluency.wav',
    'phonemic_fluency': '*_phonemic_fluency.wav'
}

# Label mappings
LABEL_MAPPINGS = {
    '2-way': {
        'HC': 0,          # Healthy Control
        'MCI': 1,         # Mild Cognitive Impairment  
        'Dementia': 1     # Dementia (grouped with MCI as "impaired")
    },
    '3-way': {
        'HC': 0,          # Healthy Control
        'MCI': 1,         # Mild Cognitive Impairment
        'Dementia': 2     # Dementia
    }
}

# Preprocessing options
ROBUST_SCALING = True
FORCE_RECOMPUTE_FEATURES = False

# Logging configuration
LOG_LEVEL = 'INFO'
SAVE_DETAILED_RESULTS = True



