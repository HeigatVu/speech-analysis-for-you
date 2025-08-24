import os
import sys
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

from sklearn.preprocessing import robust_scale
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           f1_score, precision_score, recall_score)

from tqdm import tqdm
import glob
import opensmile
import librosa

# Import your existing modules
import codes.config_classifiers as config_classifiers
import codes.classifiers as classifiers


class AcousticsAnalyzer:

    def __init__(self, base_dir: str = ".", n_jobs: int = 1):
        self.base_dir = base_dir
        self.n_jobs = n_jobs

        # Directory structure
        self.data_dir = f"{self.base_dir}/data"
        self.audio_dir = f"{self.base_dir}/audio"
        self.feats_dir = f"{self.base_dir}/feats"
        self.results_dir = f"{self.base_dir}/results"

        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.audio_dir, self.feats_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Tasks in protocols
        self.tasks = {
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
                "file_pattern": "*_Q4.wav",
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

        # Classification setup:
        self.classifiers = {
            1: {
                "name": "LR", "class": LogisticRegression, "params": {"max_iter": int(2e+10), "n_jobs": n_jobs},
            },
            2: {
                "name": "SVM", "class": SVC, "params": {"probablity": True},
            },
            3: {
                "name": "MLP", "class": MLPClassifier, "params": {"max_iter": int(2e+10)},
            },
            4: {
                "name": "KNN", "class": KNeighborsClassifier, "params": {"n_jobs": n_jobs},
            },
        }

        # Acoustic feature extraction
        self.feature_set = {
            "eGeMAPSv02": opensmile.FeatureSet.GeMAPSv02,
            "ComParE_2016": opensmile.FeatureSet.ComParE_2016,
        }

        
        def _setup_logging(self):
            log_dir = f"{self.base_dir}/logs"
            os.makedirs(log_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{log_dir}/acoustics_{timestamp}.log"

            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
            )

            self.logger = logging.getLogger(__name__)


        def _load_metadata(self) -> pd.DataFrame:
            # Load metadata and check file errors
            metadata_file = f"{self.data_dir}/metadata.csv"
            if not metadata_file.exists():
                self.logger.error(f"Metadata file not found: {metadata_file}")
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
            # Check missing cols erros
            df_metadata = pd.read_csv(metadata_file)
            required_cols = ["participant_id", "diagnosis", "ethnicity", "age", "gender", "education", "labels"]
            missing_cols = [col for col in required_cols if col not in df_metadata.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns in metadata: {missing_cols}")
                raise ValueError(f"Missing columns in metadata: {missing_cols}")

            self.logger.info(f"Loading metadata for  {len(df_metadata)}")
            self.logger.info(f"Diagnosis distribution: {metadata['diagnosis'].value_counts()}")
            return df_metadata
        

        def extract_features_for_task(self, task_name: str, feature_set_name: str, df_metadata: pd.DataFrame) -> pd.DataFrame:`` 

if __name__ == "__main__":
    pass