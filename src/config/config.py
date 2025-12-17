"""

Configuration file for classifiers

This file contains configuration parameters used across the CognoSpeak
classification experiments.
"""

# Cross-validation scoring metric
# Options: 'AUC' for ROC-AUC scoring, 'KAPPA' for Cohen's Kappa
CV_SCORER = "AUC"

# Number of folds for cross-validation
N_FOLDS = 5            

# Number of parallel
N_JOBS = 5

# Data path
BASE_DIR = f"/home2/ducvu/project/speech-analysis-for-you/"
DATA_PATH = f"/home2/ducvu/project/speech-analysis-for-you/data"
OUTPUT_PATH = f"/home2/ducvu/project/speech-analysis-for-you/output"

# Optimize classification
# Options: "simple" and "grid"
CLASS_TYPE_CHOSEN = ["simple"]

# List task in research
LIST_TASKS = {
            "Q1": {
                "name": "Maximum Phonation Time",
                "file_pattern": "Q1.wav",
            },
            "Q2": {
                "name": "Read sentences",
                "file_pattern": "Q2.wav",
            },
            "Q3": {
                "name": "Immediately recall the story",
                "file_pattern": "Q3.wav",
            },
            "Q4": {
                "name": "Picture description",
                "file_pattern": "Q4.mp3",
            },
            "Q5": {
                "name": "Picture story narrative",
                "file_pattern": "Q5.wav",
            },
            "Q6": {
                "name": "Phonetic fluency task",
                "file_pattern": "Q6.wav",
            },
            "Q7": {
                "name": "Senmantic fluency task",
                "file_pattern": "Q7.wav",
            },
            "Q8": {
                "name": "Procedural discourse task",
                "file_pattern": "Q8.wav",
            },
            "Q9": {
                "name": "Simple calculation task",
                "file_pattern": "Q9.wav",
            },
            "Q10": {
                "name": "Calling object task",
                "file_pattern": "Q10.wav",
            },
            "Q11": {
                "name": "Delayed story recall task",
                "file_pattern": "Q11.wav",
            },
}

# List chosen task
TASK_CHOSEN = ["Q4"]

LIST_ACOUSTIC = ["eGeMAPSv02", "ComParE_2016"]

# List all classification
LIST_CLASSIFIER_NAME = {
    1 : "LR",
    2 : "KNN",
    3 : "SVM",
    4 : "MLP",
    5 : "MLP_TF",   
}

# List chosen classification
CLASSIFIER_CHOSEN = [1, 3]

# Way to classification
# Options: "2-way" and "3-way"
WAY_CLASSIFICATION = ["2-way"]

# Maping label
LABEL_MAP = {
    "HC" : 0,
    "MCI" : 1,
    "Dementia": 2,
}