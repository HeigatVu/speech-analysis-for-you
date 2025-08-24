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
# import codes.classifiers as classifiers


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
        self.logger.info(f"Diagnosis distribution: {df_metadata['diagnosis'].value_counts()}")
        return df_metadata
    

    def extract_features_for_task(self, task_name: str, feature_set_name: str, df_metadata: pd.DataFrame, use_segments: bool=False) -> pd.DataFrame:
        if task_name not in self.tasks:
            self.logger.error(f"Task not found: {task_name}")
            raise ValueError(f"Task not found: {task_name}")
        
        if feature_set_name not in self.feature_set_name:
            self.logger.error(f"Feature set not found: {feature_set_name}")
            raise ValueError(f"Feature set not found: {feature_set_name}")
        
        # Choose appropriate folder for audio files
        if use_segments:
            process_audio_dir = f"{self.audio_dir}/audio_segments"
            feat_file_suffix = f"{feature_set_name}_segmented"
        else:
            process_audio_dir = self.audio_dir
            feat_file_suffix = f"{feature_set_name}_full"

        # Output path
        feat_file = f"{self.feats_dir}/{task_name}_{feat_file_suffix}.csv"

        # Check if feature does not exit
        if os.path.exists(feat_file):
            self.logger.info(f"Features already extracted for {task_name} with {feature_set_name}")
            return pd.read_csv(feat_file)

        self.logger.info(f"Extracting features for {task_name} with {feature_set_name}")

        # Initialize OpenSmile and get feature names
        smile = opensmile.Smile(feature_set=self.feature_set[feature_set_name], 
                                feature_level=opensmile.FeatureLevel.Functions,)
        
        # Create dataframe for features
        feature_names = list(smile.feature_names)
        columns = ["participant_id", "task_name"] + feature_names
        if use_segments:
            columns.insert(2, "segment_index")

        df_feat = pd.DataFrame(columns=columns)

        # Process each participant
        task_config = self.tasks[task_name]
        successful_extraction = 0

        for idx, row in tqdm(df_metadata.iterrows(), total=len(df_metadata), desc=f"Extracting features name {feature_set_name}"):
            participant_id = row["participant_id"]
            participant_dir = f"{process_audio_dir}/{participant_id}"

            # Check file patter if using segnment audio
            if use_segments:
                file_pattern = f"{participant_id}_{task_name}_seg*.wav"
            else: # Original file
                file_pattern = task_config["file_pattern"]
            
            # Find audio file for this task
            audio_files = list(participant_dir.glob(file_pattern))
            if not audio_files:
                self.logger.warning(f"No audio file for {participant_id} - {task_name} ({'segment' if use_segments else 'full audio'})")
                continue
            
            for audio_file in audio_files:
                # Extract features
                try:
                    signal, sample_rate = librosa.load(audio_file, sr=None)
                    features = smile.process_signal(signal, sample_rate)
                    features = features.reset_index(drop=True)

                    # Add metadata
                    features.insert(0, "participant_id", participant_id)
                    features.insert(1, "task_name", task_name)
                    if use_segments:
                        segment_index = int(audio_file.stem.split("_seg")[-1]) if "_seg" in audio_file.stem else 0
                        features.insert(2, "segment_index", segment_index)

                    # Append to dataframe
                    df_feat = pd.concat([df_feat, features], ignore_index=True)
                    successful_extraction += 1

                except Exception as e:
                    self.logger.error(f"Error extracting features for participant {participant_id} in task {task_name}: {e}")
                    continue
            
        if successful_extraction == 0:
            raise ValueError(f"No successful extraction for {task_name} with {feature_set_name}")
        
        # df_feat = df_feat.merge(df_metadata, on="participant_id", how="left") # Test inner

        # Save features
        df_feat.to_csv(feat_file, index=False)
        self.logger.info(f"Extracted feature for {successful_extraction} / {len(df_metadata)} participants ")
        self.logger.info(f"Feature set saved to {feat_file}")

        return df_feat
    

    def calculate_metric(self, actual_labels: np.ndarray, preds_array: np.ndarray, 
                         avg: str="macro") -> Tuple[float, float, float, np.ndarray]:   
         
        f1_val = f1_score(actual_labels, preds_array, average=avg)
        pres_val = precision_score(actual_labels, preds_array, average=avg)
        rec_val = recall_score(actual_labels, preds_array, average=avg)
        conf_val = confusion_matrix(actual_labels, preds_array)

        return f1_val, pres_val, rec_val, conf_val


    def majority_voting_pred_labels(self, df: pd.DataFrame, class_type: str="2-way", 
                         verbose: bool=False) -> Tuple[pd.DataFrame, float, float, float, np.array]:
        
        # Group  predictions by participant
        grouped_data = df.groupby("participant_id").agg({
            "pred_label": "mean",
            "labels": "first",
        })

        # Apply threshold to predictions
        if class_type == "3-way":
            threshold = 0.33333
            final_pred_labels = []
            for pred in grouped_data["pred_label"]:
                if pred < threshold:
                    final_pred_labels.append(0)
                elif pred < 2 * threshold:
                    final_pred_labels.append(1)
                else:
                    final_pred_labels.append(2)
        else: # 2-way
            threshold = 0.5
            final_pred_labels = [1 if pred >= threshold else 0 for pred in grouped_data["pred_label"]]

        grouped_data["final_pred_label"] = final_pred_labels

        # Calcualte metric
        actual_labels = grouped_data["labels"].values
        pred_labels = grouped_data["final_pred_label"].values

        f1_val, pres_val, rec_val, conf_val = self.metrics(actual_labels, pred_labels)

        if verbose:
            self.logger.info("Participant levels resuls {class_type}}")
            self.logger.info(f"F1 score: {f1_val:.4f}")
            self.logger.info(f"Precision: {pres_val:.4f}")
            self.logger.info(f"Recall: {rec_val:.4f}")
            self.logger.info(f"Confusion matrix: \n{conf_val}")

        return grouped_data, f1_val, pres_val, rec_val, conf_val
    

    def run_classification_experiment(self, task_name: str, feature_set_name: str, 
                                      classifier_ids: List[int]=[1, 3], class_type: str="2-way", 
                                      n_folds: int=5, use_segments: bool=False) -> Dict:
        self.logger.info(f"{'-'*50}")
        self.logger.info(f"Running classification experiment for {task_name} with {feature_set_name}")
        self.logger.info(f"Classification type: {class_type}")
        self.logger.info(f"Number of folds: {n_folds}")
        self.logger.info(f"Use segments: {use_segments}")

        # Load features
        df_metadata = self._load_metadata()
        df_feat = self.extract_features_for_task(task_name, feature_set_name, df_metadata, use_segments=use_segments)

        # Get feature name (exclude metadata columns)
        exclude_cols = ["participant_id", "task_name", "diagnosis", "ethnicity", "age", "gender", "education", "segment_index"]
        exclude_cols.extend([col for col in df_feat.columns if col .startswith("FOLD_")])
        feature_names = [col for col in df_feat.columns if col not in exclude_cols]

        self.logger.info(f"Using {len(feature_names)} features")

        results = {}

        for classifier_id in classifier_ids:
            if classifier_id not in self.classifiers:
                self.logger.warning(f"Classifier {classifier_id} not found")
                continue
            
            classifier_info = self.classifiers[classifier_id]
            classifier_name = classifier_info["name"]

            self.logger.info(f"Testing classifier {classifier_name}...")

            fold_results = {
                "f1_score": [],
                "precision": [],
                "recall": [],
                "confusion_matrix": [],
            }

            # Cross-validation
            for fold in range(n_folds):
                fold_col = f"FOLD_{fold}"
                if fold_col not in df_feat.columns:
                    self.logger.warning(f"Fold {fold} not found in features")
                    continue

                # Split data into train and test
                df_train = df_feat[df_feat[fold_col] == "TRAIN"]
                df_test = df_feat[df_feat[fold_col] == "TEST"]

                if len(df_train) == 0 or len(df_test) == 0:
                    self.logger.warning(f"Empty train or test set for fold {fold}")
                    continue

                # Prepare features and labels
                x_train = robust_scale(np.array(df_train[feature_names])) # Normalize features
                y_train = df_train["labels"].values
                x_test = robust_scale(np.array(df_test[feature_names])) # Normalize features
                y_test = df_test["labels"].values

                # Train classifier
                try:
                    classifier = classifier_info["class"](**classifier_info["params"])
                    classifier.fit(x_train, y_train)

                    # Predict
                    y_preds = classifier.predict(x_test)

                    # Create results
                    df_results = df_test[["participant_id", "labels"]].copy()
                    df_results["pred_label"] = y_preds

                    # Voting
                    _, f1_val, pres_val, rec_val, conf_val = self.majority_voting_pred_labels(df_results, class_type=class_type)

                    fold_results["f1_score"].append(f1_val)
                    fold_results["precision"].append(pres_val)
                    fold_results["recall"].append(rec_val)
                    fold_results["confusion_matrix"].append(conf_val)

                except Exception as e:
                    self.logger.error(f"Error in fold {fold} for {classifier_name}: {e}")
                    raise e
                    continue

            # Summarize results
            if fold_results['f1_scores']:
                mean_f1 = np.mean(fold_results['f1_scores'])
                std_f1 = np.std(fold_results['f1_scores'])
                mean_prec = np.mean(fold_results['precision_scores'])
                std_prec = np.std(fold_results['precision_scores'])
                mean_rec = np.mean(fold_results['recall_scores'])
                std_rec = np.std(fold_results['recall_scores'])
                
                self.logger.info(f"{classifier_name} Results:")
                self.logger.info(f"  F1-score: {mean_f1:.3f} ± {std_f1:.3f}")
                self.logger.info(f"  Precision: {mean_prec:.3f} ± {std_prec:.3f}")
                self.logger.info(f"  Recall: {mean_rec:.3f} ± {std_rec:.3f}")
                self.logger.info(f"  Best F1: {max(fold_results['f1_scores']):.3f}")
                
                results[classifier_name] = {
                    'mean_f1': mean_f1,
                    'std_f1': std_f1,
                    'mean_precision': mean_prec,
                    'std_precision': std_prec,
                    'mean_recall': mean_rec,
                    'std_recall': std_rec,
                    'best_f1': max(fold_results['f1_scores']),
                    'all_f1_scores': fold_results['f1_scores'],
                    'fold_details': fold_results['fold_details']
                }
        return results
    

    def save_results(self, results: Dict, task_name: str, feature_set_name: str):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"{self.results_dir}/{task_name}_{feature_set_name}_{timestamp}.csv"

        # Convert reesult into df
        rows = []
        for classifier_name, metrics in results.items():
            row = {
                'task': task_name,
                'feature_set': feature_set_name,
                'classifier': classifier_name,
                'mean_f1': metrics['mean_f1'],
                'std_f1': metrics['std_f1'],
                'mean_precision': metrics['mean_precision'],
                'std_precision': metrics['std_precision'],
                'mean_recall': metrics['mean_recall'],
                'std_recall': metrics['std_recall'],
                'best_f1': metrics['best_f1'],
                'timestamp': timestamp
            }
            rows.append(row)

        df_results = pd.DataFrame(rows)
        df_results.to_csv(result_file, index=False)
        self.logger.info(f"Results saved to {result_file}")



if __name__ == "__main__":
    pass