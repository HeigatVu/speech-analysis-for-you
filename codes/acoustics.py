import os
import sys
import datetime
import pandas as pd
import numpy as np

from sklearn.preprocessing import binarize, scale, robust_scale
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, r2_score, mean_squared_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tqdm import tqdm
import glob
import codes.config_classifiers as config_classifiers
import classifiers
import opensmile, librosa


def read_file(file):
    f = open(file, 'r')
    return f.read()

def calc_metrics(actual_labels, pred_vals, avg=None):
    if avg == None:
        f1_val = f1_score(actual_labels, pred_vals)
        pres_val = precision_score(actual_labels, pred_vals)
        rec_val = recall_score(actual_labels, pred_vals)
    
    else:
        f1_val = f1_score(actual_labels, pred_vals, average=avg)
        pres_val = precision_score(actual_labels, pred_vals, average=avg)
        rec_val = recall_score(actual_labels, pred_vals, average=avg)
        
    conf_val = confusion_matrix(actual_labels, pred_vals)
    return f1_val, pres_val, rec_val, conf_val


def majority_voting_pred_labels(df, a_class_type, verbose):
    data = {
        "r_IDs": df[["r_IDs", "pred_label"]].groupby("r_IDs").mean().index.values,
        "grouped_pred_label": df[["r_IDs", "pred_label"]].groupby("r_IDs").mean().pred_label.values
    }
    df_final_test_grouped = df.merge(pd.DataFrame(data), on="r_IDs", how="inner")

    df_final_test_grouped = df_final_test_grouped.drop_duplicates(subset=["r_IDs"], keep="first")

    if a_class_type =="3-way":
        threshold_val = 0.333333

        final_pred_label_list = []
        for x in df_final_test_grouped.grouped_pred_label:
            if x < threshold_val:
                final_pred_label_list.append(0)
            elif x >= threshold_val and x < (2*threshold_val):
                final_pred_label_list.append(1)
            else:
                final_pred_label_list.append(2)

    else:
        threshold_val = 0.5

        final_pred_label_list = []
        for x in df_final_test_grouped.grouped_pred_label:
            if x < threshold_val:
                final_pred_label_list.append(0)
            else:
                final_pred_label_list.append(1)

    df_final_test_grouped.insert(len(df_final_test_grouped.columns), 
                                 "final_pred_label", final_pred_label_list)
    
    actual_labels = df_final_test_grouped.labels

    col_2_cons = "final_pred_label"
    pred_vals = df_final_test_grouped[col_2_cons]

    f1_val, pres_val, rec_val, conf_val = calc_metrics(actual_labels, pred_vals, avg="macro")


    if verbose == 1:
        print("Macro F1-score:", round(f1_val, 2))
        print("Macro Precision:", round(pres_val, 2))
        print("Macro Recall:", round(rec_val, 2))
        print("Confusion Matrix:\n", conf_val)

    if a_class_type == "3-way":
        precision, recall, fscore, support = score(actual_labels, pred_vals)
        if verbose == 1:
            print("Metric \t HC \t MCI \t Dementia")
            print(f"Precision \t {precision[0]:.2f} \t {precision[1]:.2f} \t {precision[2]:.2f}")
            print(f"Recall \t {recall[0]:.2f} \t {recall[1]:.2f} \t {recall[2]:.2f}")
            print(f"F-score \t {fscore[0]:.2f} \t {fscore[1]:.2f} \t {fscore[2]:.2f}")
            print(f"Support \t {support[0]:.2f} \t {support[1]:.2f} \t {support[2]:.2f}")

    return df_final_test_grouped, f1_val, pres_val, rec_val, conf_val

if __name__ == "__main__" and "__file__" in globals():

    start_time = datetime.now()
    current_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n\n CognoSpeak Acoustics - {current_time} \n\n")

    N_jobs = int(sys.argv[1])
    '''
    For debugging ........ 
    classifier_idx = 1 # LR
    classifier_idx = 2 # KNN
    classifier_idx = 3 # SVM
    classifier_idx = 4 # MLP
    classifier_idx = 5 # MLP_TF
    
    N_jobs = 5
    
    '''

    CV_SCORER = config_classifiers.CV_SCORER
    N_FOLDS = config_classifiers.N_FOLDS

    print(f"CV_SCORER: {CV_SCORER}")
    print(f"N_FOLDS: {N_FOLDS}")
    print(f"Number of jobs: {N_jobs}")

    BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    FEAT_DIR = os.path.join(BASE_DIR, "feats")
    RESULT_DIR = os.path.join(BASE_DIR, "results")
    

    # # Read metadata and staistic independent variables
    # df_meta_final = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

    # # Count the number of samples for each diagnosis (HC, MCI, Dementia)
    # df_meta_final.diagnosis.value_counts()

    # # Age distribution
    # np.mean(df_meta_final[df_meta_final.diagnosis == 'Dementia'].age)
    # np.std(df_meta_final[df_meta_final.diagnosis == 'Dementia'].age)
    # np.mean(df_meta_final[df_meta_final.diagnosis == 'MCI'].age)
    # np.std(df_meta_final[df_meta_final.diagnosis == 'MCI'].age)
    # np.mean(df_meta_final[df_meta_final.diagnosis == 'HC'].age)
    # np.std(df_meta_final[df_meta_final.diagnosis == 'HC'].age)
    
    # np.mean(df_meta_final.age)
    # np.std(df_meta_final.age)

    # bin_range = (0, 50, 60, 70, 80, 100)
    # np.histogram(df_meta_final[df_meta_final.diagnosis == 'Dementia'].age, bins=bin_range)[0]
    # np.histogram(df_meta_final[df_meta_final.diagnosis == 'MCI'].age, bins=bin_range)[0]
    # np.histogram(df_meta_final[df_meta_final.diagnosis == 'HC'].age, bins=bin_range)[0]

    # # Ethgnicity distribution
    # df_meta_final.ethnicity.value_counts()
    
    # df_ethnic = df_meta_final.replace(to_replace=["Chinese", "Pakistani"],
    #        value="Asian")
    # df_ethnic = df_ethnic.replace(to_replace=["Undisclosed"],
    #        value="Unknown")
    # df_ethnic = df_ethnic.replace(to_replace=["English, Welsh, Scottish, Northern Irish or British"],
    #        value="White British")
    # df_ethnic = df_ethnic.replace(to_replace=["White Irish"],
    #        value="Other White")
    # df_ethnic = df_ethnic.replace(to_replace=["Black Caribbean"],
    #        value="Mixed")
    # df_ethnic.ethnicity.value_counts()
    
    # df_ethnic[df_ethnic.diagnosis == 'Dementia'].ethnicity.value_counts()
    # df_ethnic[df_ethnic.diagnosis == 'MCI'].ethnicity.value_counts()
    # df_ethnic[df_ethnic.diagnosis == 'HC'].ethnicity.value_counts()
    
    # for an_eth in list(set(df_ethnic.ethnicity)):
    #     print('')
    #     print('Ethnicity : ', an_eth)
    #     print(df_ethnic[df_ethnic.ethnicity == an_eth].diagnosis.value_counts())
    #     print('')

    # # Gender distribution 
    # df_meta_final.gender.value_counts()
    # df_meta_final[df_meta_final.diagnosis == 'Dementia'].gender.value_counts()
    # df_meta_final[df_meta_final.diagnosis == 'MCI'].gender.value_counts()
    # df_meta_final[df_meta_final.diagnosis == 'HC'].gender.value_counts()


    l_q_types = ["Q4", "Q6", "Q10", "Q12", "ALL"]
    l_feat_types = ["eGeMAPSv02", "ComParE_2016"]

    # l_class_ways = ['2-way', '3-way']
    l_class_ways = ['2-way']
    
    class_idx_lists = [1, 3]
    
    # Add regress here for regression values 
    # list_class_type = ['simple', 'grid']
    list_class_type = ['simple']

    for classifier_idx in class_idx_lists:
        if classifier_idx == 1:
            classifier_name = "LR"
        
        elif classifier_idx == 2:
            classifier_name = 'KNN'
        
        elif classifier_idx == 3:
            classifier_name = 'SVM'
        
        elif classifier_idx == 4:
            classifier_name = 'MLP'
        
        elif classifier_idx == 5:
            classifier_name = 'MLP_TF'

        for Q_2_consider in l_q_types:
            for class_type in list_class_type:
                for openSmile_feat in l_feat_types:
                    for a_class_type in l_class_ways:
                        print("\n--------------------------------")
                        print(f"Question selected is: {Q_2_consider}")
                        print(f"Classifier selected is: {classifier_name}")
                        print(f"Hyperparameter optimisation type is: {class_type}")

                        df_feat_name = os.path.join(BASE_DIR, "feats/openSmile_feat.csv")

                        if os.path.exists(df_feat_name):
                            df_feat = pd.read_csv(df_feat_name)
                        else:
                            if openSmile_feat == "ComParE_2016" or openSmile_feat == "eGeMAPSv02":
                                if openSmile_feat == "ComParE_2016":
                                    smile = opensmile.Smile(
                                        feature_set=opensmile.FeatureSet.ComParE_2016,
                                        feature_level=opensmile.FeatureLevel.Functionals,
                                    )
                                elif openSmile_feat == "eGeMAPSv02":
                                    smile = opensmile.Smile(
                                        feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                        feature_level=opensmile.FeatureLevel.Functionals,
                                    )
                                df_feat = pd.DataFrame([], columns=["dir_name", "Q_type"]+smile.feature_names)
                                
                                df_meta_final = df_meta_final.reset_index(drop=True)
                                for ind in tqdm(df_meta_final.index):
                                    for q_type in l_q_types:
                                        if q_type == "ALL":
                                            continue
                                        audio_pattern = os.path.join(DATA_DIR, df_meta_final['dir_name'][ind], 
                                                                    f"{df_meta_final['dir_name'][ind]}*_{q_type}*.wav")
                                        an_audio = glob.glob(audio_pattern)
                                        
                                        sig,sample_rate = librosa.load(an_audio, sr=None)
                                        df_os_feat = smile.process_signal(sig, sample_rate)
                                        df_os_feat = df_os_feat.reset_index(drop=True)

                                        df_os_feat.insert(0, "dir_name", df_meta_final["dir_name"][ind])
                                        df_os_feat.insert(1, "Q_type", q_type)

                                        df_feat = pd.concat([df_feat, df_os_feat], ignore_idex=True, sort=False)       

                                df_feat = df_feat.merge(df_meta_final, on='dir_name', how='inner')
                                # df_feat = df_feat.rename(columns={'anyon_IDs': 'r_IDs'})
                                df_feat.to_csv(df_feat_name, index=False)

                            if openSmile_feat == 'ComParE_2016' or openSmile_feat == 'eGeMAPSv02':
                        
                                if openSmile_feat == 'ComParE_2016':
                                    smile = opensmile.Smile(
                                        feature_set=opensmile.FeatureSet.ComParE_2016,
                                        feature_level=opensmile.FeatureLevel.Functionals,)
                                elif openSmile_feat == 'eGeMAPSv02':
                                    smile = opensmile.Smile(
                                        feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                        feature_level=opensmile.FeatureLevel.Functionals,)
                            
                                feat_names = list(smile.feature_names)

                            print(f"Feature extration is dont for {openSmile_feat}")
                            print(f"This is a {a_class_type} type classification")


                            list_f1_vals = []
                            list_pres = []
                            list_recalls = []

                            for k in range(N_FOLDS):
                                if Q_2_consider == "ALL":
                                    df_train = df_feat[df_feat[f"FOLD_{str(k)}"] == "TRAIN"]
                                    df_test = df_feat[df_feat[f"FOLD_{str(k)}"] == "TEST"]
                                else:
                                    df_train = df_feat[
                                        (df_feat[f"FOLD_{str(k)}"] == "TRAIN") & (df.feat.Q_type==Q_2_consider)
                                    ]
                                    df_test = df_feat[
                                        (df_feat[f"FOLD_{str(k)}"] == "TEST") & (df.feat.Q_type==Q_2_consider)
                                    ]

                                try:
                                    if class_type == "grid":
                                        if classifier_idx == 1:
                                            if a_class_type == "3-way":
                                                opt_LRM, to_save_params = classifiers.cv_param_estimation_LR_multiclass(df_train, feat_names, CV_SCORER, N_jobs)
                                            else:
                                                opt_LRM, to_save_params = classifiers.cv_param_estimation_LR(df_train, feat_names, CV_SCORER, N_jobs)
                                        elif classifier_idx == 2:
                                            opt_KNN, to_save_params = classifiers.cv_param_estimation_KNN(df_train, feat_names, CV_SCORER, N_jobs)
                                        elif classifier_idx == 3:
                                            if a_class_type == "3-way":
                                                opt_LRM, to_save_params = classifiers.cv_param_estimation_SVM_multiclass(df_train, feat_names, CV_SCORER, N_jobs)
                                            else:
                                                opt_LRM, to_save_params = classifiers.cv_param_estimation_SVM(df_train, feat_names, CV_SCORER, N_jobs)
                                        elif classifier_idx == 4:
                                            if a_class_type == "3-way":
                                                opt_LRM, to_save_params = classifiers.cv_param_estimation_MLP_multiclass(df_train, feat_names, CV_SCORER, N_jobs)
                                            else:
                                                opt_LRM, to_save_params = classifiers.cv_param_estimation_MLP(df_train, feat_names, CV_SCORER, N_jobs)
                                        elif classifier_idx == 5:
                                            opt_LRM, to_save_params = classifiers.cv_param_estimation_MLP_TF(df_train, feat_names, CV_SCORER, N_jobs)
                                        trained_LRM = opt_LRM.fit(robust_scale(df_train[feat_names]), list(df_train.labels.values))
                                    elif class_type == "simple":
                                        if classifier_idx == 1:
                                            trained_LRM = LogisticRegression(max_iter=int(1e+20), n_jobs=N_jobs)
                                            trained_LRM.fit(robust_scale(np.array(df_train[feat_names])), list(df_train.labels.values))
                                        elif classifier_idx == 2:
                                            trained_KNN = KNeighborsClassifier(n_jobs=N_jobs)
                                            trained_KNN.fit(robust_scale(np.array(df_train[feat_names])), list(df_train.labels.values))
                                        elif classifier_idx == 3:
                                            trained_SVM = SVC(kernel='rbf', C=1.0, gamma=0.1, probability=True, n_jobs=N_jobs)
                                            trained_SVM.fit(robust_scale(np.array(df_train[feat_names])), list(df_train.labels.values))
                                        elif classifier_idx == 4:
                                            trained_MLP = MLPClassifier(max_iter=int(1e+20), n_jobs=N_jobs)
                                            trained_MLP.fit(robust_scale(np.array(df_train[feat_names])), list(df_train.labels.values))
                                    else:
                                        print("PLEASE INPUT GRID OR SIMPLE OR REGREE")
                                except:
                                    print(f"Error in parameter estimation for {classifier_name} with {class_type} type")

                                test_preds = trained_LRM.predict(robust_scale(np.array(df_test[feat_names])))

                                df_final_test = df_test[["dir_name", "labels"]]
                                df_final_test.insert(len(df_final_test.columns), "pred_label", test_preds)
                                df_final_test = df_final_test.rename(columns={"dir_name": "r_IDs"})

                                df_final_test, f1_val, pres_val, rec_val, conf_val = majority_voting_pred_labels(df_final_test, a_class_type, verbose=0)

                                list_f1_vals.append(f1_val)
                                list_pres.append(pres_val)
                                list_recalls.append(rec_val)

                            print('List of F1-scores: ', list_f1_vals)
                            print('Max F1-score: ', max(list_f1_vals))
                            print('Mean F1-score: ', round(np.mean(list_f1_vals), 2), ' with STD : ', round(np.std(list_f1_vals), 2))
                            print('Mean Precision: ', round(np.mean(list_pres), 2), ' with STD : ', round(np.std(list_pres), 2))
                            print('Mean Recall: ', round(np.mean(list_recalls), 2), ' with STD : ', round(np.std(list_recalls), 2))
                                
                            print('----------------------')
                            sys.stdout.flush()

    exceutionTime = datetime.now() - start_time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n\n CognoSpeak Acoustics completed at: {str(current_time)} \n\n")
    print(f"Execution time: {str(exceutionTime)}")