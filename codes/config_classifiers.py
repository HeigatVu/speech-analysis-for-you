"""

Configuration file for classifiers

This file contains configuration parameters used across the CognoSpeak
classification experiments.
"""

# Cross-validation scoring metric
# Options: 'AUC' for ROC-AUC scoring, 'KAPPA' for Cohen's Kappa
CV_SCORER = "AUC"

# Number of folds for cross-validation
# The paper mentions k=5 for 5-fold cross-validation
N_FOLDS = 5            

# Data path
BASE_DIR = "~/cogno-speak"
DATA_PATH = f"/mnt/data_lab513/ducvu/fake-speech-data"
FEATS_PATH = f"{BASE_DIR}/feats/"
RESULTS_PATH = f"{BASE_DIR}/results/"

LIST_TASKS = {
            "Q1": {
                "name": "Maximum Phonation Time",
                "file_pattern": "*_Q1.wav",
            },
            "Q2": {
                "name": "Read sentences",
                "file_pattern": "*_Q2.wav",
            },
            "Q3": {
                "name": "Immediately recall the story",
                "file_pattern": "*_Q3.wav",
            },
            "Q4": {
                "name": "Picture description",
                "file_pattern": "*_Q4.mp3",
            },
            "Q5": {
                "name": "Recall picture story narrative",
                "file_pattern": "*_Q5.wav",
            },
            "Q6": {
                "name": "Phonetic fluency task",
                "file_pattern": "*_Q6.wav",
            },
            "Q7": {
                "name": "Senmantic fluency task",
                "file_pattern": "*_Q7.wav",
            },
            "Q8": {
                "name": "Procedural discourse",
                "file_pattern": "*_Q8.wav",
            },
            "Q9": {
                "name": "Simple calculation",
                "file_pattern": "*_Q9.wav",
            },
            "Q10": {
                "name": "Object recall task",
                "file_pattern": "*_Q9.wav",
            },
            "Q11": {
                "name": "Delayed story recall",
                "file_pattern": "*_Q10.wav",
            },
}

LIST_ACOUSTIC_TYPE = ["eGeMAPSv02", "ComParE_2016"]

LIST_CLASSIFIER_NAME = {
    1 : "LR",
    2 : "KNN",
    3 : "SVM",
    4 : "MLP",
    5 : "MLP_TF",   
}