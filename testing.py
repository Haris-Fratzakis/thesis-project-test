import os
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from datetime import datetime

import feature_extraction as feat_extr
import training

# Adding a directory choice to manage the name of each dataset
# Dataset directory choice
data_dir_choice = "smarty4covid"

# IMPORTANT, if run on different PCs, this needs to be changed to point to the dataset directory
# Every dataset path query is formed in relation to this variable (data_dir)
# Load the index csv
if data_dir_choice == "smarty4covid":
    data_dir = "E:/Storage/University/Thesis/smarty4covid/"
elif data_dir_choice == "coswara":
    data_dir = "E:/Storage/University/Thesis/iiscleap-Coswara-Data-bf300ae/Extracted_data_mix"
else:
    print("No Data Specified, thus defaulting to smarty4covid")
    data_dir = "E:/Storage/University/Thesis/smarty4covid/"


# New Test function to have modular feature extraction and training functions
def test_modular():
    # Load the index csv
    if data_dir_choice == "smarty4covid":
        data_index = os.path.join(data_dir, 'smarty4covid_tabular_data.csv')
        data = pd.read_csv(data_index)

        # Exclude rows where 'covid_status' is 'no'
        data = data[data.covid_status != 'no']
    elif data_dir_choice == "coswara":
        data_index = os.path.join(data_dir, 'combined_data_renamed.csv')
        data = pd.read_csv(data_index)

        # Exclude rows where 'covid_status' is 'under validation' or 'resp_illness_not_identified'
        data = data[data.covid_status != 'under_validation']
        data = data[data.covid_status != 'resp_illness_not_identified']
    else:
        # No Data Specified, thus defaulting to smarty4covid
        data_index = os.path.join(data_dir, 'smarty4covid_tabular_data.csv')
        data = pd.read_csv(data_index)

        # Exclude rows where 'covid_status' is 'no'
        data = data[data.covid_status != 'no']

    # This is the modular feature extraction stage

    # Hyperparameter values based on:
    # Preliminary diagnosis of COVID-19 based on cough sounds using machine learning algorithms
    # https://ieeexplore.ieee.org/abstract/document/9432324
    # k_values_mfcc = [1, 2, 3, 4, 5]
    # k_values_frame = [8, 9, 10, 11, 12]
    # k_values_segment = [5, 7, 10, 12, 15]

    # Test
    k_values_mfcc = [1, 2, 3, 4, 5]
    k_values_frame = [8, 9, 10, 11, 12]
    k_values_segment = [7, 10]
    test_feat_extr(data=data, k_values_mfcc=k_values_mfcc, k_values_frame=k_values_frame,
                   k_values_segment=k_values_segment)

    # models_used signifies which model is used, each slot signifies a different model
    # 1 means model is going to be used, 0 means it will not be used
    # slots are [LR, KNN, SVM, MLP, CNN, LSTM, TestUsageModel]
    models_used = [1, 1, 0, 0, 0, 0, 0]
    test_size = [0.4]
    # TODO Fix SVM bug of never converging to a solution
    # This is the modular classifier training stage
    results_df, parameters_df = test_classifier_mod(k_values_mfcc=k_values_mfcc, k_values_frame=k_values_frame,
                                     k_values_segment=k_values_segment, models_used=models_used, test_size=test_size)
    # print(results_df)

    # convert models_used values to their names
    for idx in range(len(models_used)):
        if models_used[idx] == 1:
            models_used[idx] = models_name_conv(idx)

    print("Models Used List: " + str(models_used))
    test_display(results_df, models_used, parameters_df)

    return results_df


# Making sure all features vectors are the same shape
def pad_or_truncate(features, target_length):
    if len(features) > target_length:
        return features[:target_length]
    elif len(features) < target_length:
        return np.pad(features, (0, target_length - len(features)), 'constant')
    else:
        return features


# Temporary feature extraction function
def test_feat_extr(data, k_values_mfcc=None, k_values_frame=None, k_values_segment=None):
    if k_values_mfcc is None:
        k_values_mfcc = [1]
    if k_values_frame is None:
        k_values_frame = [-1]
    if k_values_segment is None:
        k_values_segment = [-1]

    # Initialize the LabelEncoder
    le = LabelEncoder()

    # Loop over all combinations of hyperparameters
    for k_mfcc in k_values_mfcc:
        n_mfcc = 14 * k_mfcc

        # Feature Extraction Initialization Function with only one method and only one hyperparameter
        if k_values_frame == [-1]:
            # Name of the directory and file where the features will be saved
            features_folder = data_dir_choice + "/extracted_features/feat_extr_simple"

            # Check if the directory exists, if not, create it
            if not os.path.exists(features_folder):
                os.makedirs(features_folder)

            feature_filename_target = "extracted_features_" + str(k_mfcc) + ".npy"
            label_filename_target = "extracted_labels_" + str(k_mfcc) + ".npy"
            feature_filename = os.path.join(features_folder, feature_filename_target)
            label_filename = os.path.join(features_folder, label_filename_target)

            successful_indices = []
            features_list = []
            # Check if the file doesn't exist
            if not os.path.exists(feature_filename):
                # Modified part to extract features and simultaneously filter labels
                for idx, row in data.iterrows():
                    if data_dir_choice == "smarty4covid":
                        path_part_1 = row.participantid
                        path_part_2 = row.submissionid
                        audio_path = os.path.join(path_part_1, path_part_2)
                        audio_name = "audio.cough.mp3"
                    elif data_dir_choice == "coswara":
                        audio_path = row.id
                        audio_name = "cough-heavy.wav"
                    else:
                        path_part_1 = row.participantid
                        path_part_2 = row.submissionid
                        audio_path = os.path.join(path_part_1, path_part_2)
                        audio_name = "audio.cough.mp3"
                    feat = feat_extr.extract_features_simple(data_dir, audio_path, audio_name, n_mfcc)
                    if feat is not False:
                        features_list.append(feat)
                        successful_indices.append(idx)
                features = np.array(features_list)

                # Filter labels based on successful feature extraction
                labels = np.array(data.loc[successful_indices, 'covid_status'])

                # Convert labels to a consistent numerical format
                labels = le.fit_transform(labels)

                np.save(feature_filename, features)
                np.save(label_filename, labels)
        else:
            for k_frame in k_values_frame:
                frame_size = 2 ** k_frame
                hop_length = frame_size // 2  # 50% overlap

                # Feature Extraction Initialization Function with four methods and only two hyperparameters
                if k_values_segment == [-1]:
                    # Name of the directory and file where the features will be saved
                    features_folder = data_dir_choice + "/extracted_features/feat_extr"

                    # Check if the directory exists, if not, create it
                    if not os.path.exists(features_folder):
                        os.makedirs(features_folder)

                    feature_filename_target = "extracted_features_" + str(k_mfcc) + "_" + str(k_frame) + ".npy"
                    label_filename_target = "extracted_labels_" + str(k_mfcc) + "_" + str(k_frame) + ".npy"
                    feature_filename = os.path.join(features_folder, feature_filename_target)
                    label_filename = os.path.join(features_folder, label_filename_target)

                    successful_indices = []
                    features_list = []
                    # Check if the file doesn't exist
                    if not os.path.exists(feature_filename):
                        # Modified part to extract features and simultaneously filter labels
                        for idx, row in data.iterrows():
                            if data_dir_choice == "smarty4covid":
                                path_part_1 = row.participantid
                                path_part_2 = row.submissionid
                                audio_path = os.path.join(path_part_1, path_part_2)
                                audio_name = "audio.cough.mp3"
                            elif data_dir_choice == "coswara":
                                audio_path = row.id
                                audio_name = "cough-heavy.wav"
                            else:
                                path_part_1 = row.participantid
                                path_part_2 = row.submissionid
                                audio_path = os.path.join(path_part_1, path_part_2)
                                audio_name = "audio.cough.mp3"
                            feat = feat_extr.extract_features(data_dir, audio_path, audio_name, n_mfcc, frame_size,
                                                              hop_length)
                            if feat is not False:
                                features_list.append(feat)
                                successful_indices.append(idx)
                        features = np.array(features_list)

                        # Filter labels based on successful feature extraction
                        labels = np.array(data.loc[successful_indices, 'covid_status'])
                        # Convert labels to a consistent numerical format
                        labels = le.fit_transform(labels)

                        np.save(feature_filename, features)
                        np.save(label_filename, labels)
                else:
                    for k_segment in k_values_segment:
                        n_segments = 10 * k_segment

                        # Feature Extraction Initialization Function with all five methods and all three hyperparameters

                        # Name of the directory and file where the features will be saved
                        features_folder = data_dir_choice + "/extracted_features/feat_extr_with_segm"

                        # Check if the directory exists, if not, create it
                        if not os.path.exists(features_folder):
                            os.makedirs(features_folder)

                        feature_filename_target = "extracted_features_" + str(k_mfcc) + "_" + str(k_frame) + "_" + str(
                            k_segment) + ".npy"
                        label_filename_target = "extracted_labels_" + str(k_mfcc) + "_" + str(k_frame) + "_" + str(
                            k_segment) + ".npy"
                        feature_filename = os.path.join(features_folder, feature_filename_target)
                        label_filename = os.path.join(features_folder, label_filename_target)

                        successful_indices = []
                        features_list = []
                        # Check if the file doesn't exist
                        if not os.path.exists(feature_filename):
                            # Modified part to extract features and simultaneously filter labels
                            for idx, row in data.iterrows():
                                if data_dir_choice == "smarty4covid":
                                    path_part_1 = row.participantid
                                    path_part_2 = row.submissionid
                                    audio_path = os.path.join(path_part_1, path_part_2)
                                    audio_name = "audio.cough.mp3"
                                elif data_dir_choice == "coswara":
                                    audio_path = row.id
                                    audio_name = "cough-heavy.wav"
                                else:
                                    path_part_1 = row.participantid
                                    path_part_2 = row.submissionid
                                    audio_path = os.path.join(path_part_1, path_part_2)
                                    audio_name = "audio.cough.mp3"
                                feat = feat_extr.extract_features_with_segments(data_dir, audio_path, audio_name,
                                                                                n_mfcc, frame_size, hop_length,
                                                                                n_segments)
                                if feat is not False:
                                    features_list.append(feat)
                                    successful_indices.append(idx)

                            # check if all features have the same shape
                            shapes = [f.shape for f in features_list]
                            unique_shapes = set(shapes)
                            for shape in unique_shapes:
                                print(f"Shape: {shape}, Count: {shapes.count(shape)}")
                            if len(unique_shapes) > 1:
                                print("Inconsistent shapes found in features_list:")
                                for shape in unique_shapes:
                                    print(f"Shape: {shape}, Count: {shapes.count(shape)}")

                                lengths = [len(f) for f in features_list]
                                print(lengths)

                                # if there are different size vectors
                                most_common_length_result = mode(lengths)
                                if isinstance(most_common_length_result.mode, np.ndarray):
                                    most_common_length = most_common_length_result.mode[0]
                                else:
                                    most_common_length = most_common_length_result.mode

                                # Shaping all features vectors to use the most common length
                                target_length = most_common_length
                                features_list = [pad_or_truncate(f, target_length) for f in features_list]
                            features = np.array(features_list)

                            # Filter labels based on successful feature extraction
                            labels = np.array(data.loc[successful_indices, 'covid_status'])

                            # Convert labels to a consistent numerical format
                            labels = le.fit_transform(labels)

                            np.save(feature_filename, features)
                            np.save(label_filename, labels)


# Temporary classifier function
def test_classifier_mod(k_values_mfcc, k_values_frame=None, k_values_segment=None, models_used=None, test_size=None):
    if k_values_mfcc is None:
        k_values_mfcc = [1]
    if k_values_frame is None:
        k_values_frame = [-1]
    if k_values_segment is None:
        k_values_segment = [-1]
    if models_used is None:
        models_used = [0, 0, 0, 0, 0, 0]
    if test_size is None:
        k_values_segment = [0.3]

    # Initialize list for storing results
    results = []
    parameters = []

    total_iterations = len(k_values_mfcc) * len(k_values_frame) * len(k_values_segment) * len(test_size)
    current_iteration = 0
    # Loop over all combinations of hyperparameters
    for test_size_val in test_size:
        for k_mfcc in k_values_mfcc:
            n_mfcc = 14 * k_mfcc

            # Feature Extraction Initialization Function with only one method and only one hyperparameter
            if k_values_frame == [-1]:
                # Name of the directory and file where the features will be saved
                features_folder = data_dir_choice + "/extracted_features/feat_extr_simple"

                # Check if the directory exists, if not, report error, data mismatch
                if not os.path.exists(features_folder):
                    print("Error, data mismatch, features folder doesn't exist in test classifier")

                feature_filename_target = "extracted_features_" + str(k_mfcc) + ".npy"
                label_filename_target = "extracted_labels_" + str(k_mfcc) + ".npy"
                feature_filename = os.path.join(features_folder, feature_filename_target)
                label_filename = os.path.join(features_folder, label_filename_target)

                # Check if the file doesn't exist
                if os.path.exists(feature_filename):
                    features = np.load(feature_filename)
                    labels = np.load(label_filename)

                    # Train and evaluate the different classifiers outlined in training.py
                    current_iteration += 1
                    print("Iteration " + str(current_iteration) + "/" + str(total_iterations))
                    run_res, run_param = test_classifier(features, labels, n_mfcc=n_mfcc, models_used=models_used, test_size=test_size_val)
                    results.append(run_res)
                    parameters.append(run_param)
                else:
                    print(
                        "Error, data mismatch, features and labels data don't exist in features folder in test classifier")
            else:
                for k_frame in k_values_frame:
                    frame_size = 2 ** k_frame
                    hop_length = frame_size // 2  # 50% overlap

                    # Feature Extraction Initialization Function with all five methods and only two hyperparameters
                    if k_values_segment == [-1]:
                        # Name of the directory and file where the features will be saved
                        features_folder = data_dir_choice + "/extracted_features/feat_extr"

                        # Check if the directory exists, if not, report error, data mismatch
                        if not os.path.exists(features_folder):
                            print("Error, data mismatch, features folder doesn't exist in test classifier")

                        feature_filename_target = "extracted_features_" + str(k_mfcc) + "_" + str(k_frame) + ".npy"
                        label_filename_target = "extracted_labels_" + str(k_mfcc) + "_" + str(k_frame) + ".npy"
                        feature_filename = os.path.join(features_folder, feature_filename_target)
                        label_filename = os.path.join(features_folder, label_filename_target)

                        # Check if the file doesn't exist
                        if os.path.exists(feature_filename):
                            features = np.load(feature_filename)
                            labels = np.load(label_filename)

                            # Train and evaluate the different classifiers outlined in training.py
                            current_iteration += 1
                            print("Iteration " + str(current_iteration) + "/" + str(total_iterations))
                            run_res, run_param = test_classifier(features, labels, n_mfcc=n_mfcc, frame_size=frame_size,
                                                                 models_used=models_used, test_size=test_size_val)
                            results.append(run_res)
                            parameters.append(run_param)
                        else:
                            print(
                                "Error, data mismatch, features and labels data don't exist in features folder in test classifier")
                    else:
                        for k_segment in k_values_segment:
                            n_segments = 10 * k_segment

                            # Feature Extraction Initialization Function with all five methods and all three hyperparameters

                            # Name of the directory and file where the features will be saved
                            features_folder = data_dir_choice + "/extracted_features/feat_extr_with_segm"

                            # Check if the directory exists, if not, report error, data mismatch
                            if not os.path.exists(features_folder):
                                print("Error, data mismatch, features folder doesn't exist in test classifier")

                            feature_filename_target = "extracted_features_" + str(k_mfcc) + "_" + str(k_frame) + "_" + str(
                                k_segment) + ".npy"
                            label_filename_target = "extracted_labels_" + str(k_mfcc) + "_" + str(k_frame) + "_" + str(
                                k_segment) + ".npy"
                            feature_filename = os.path.join(features_folder, feature_filename_target)
                            label_filename = os.path.join(features_folder, label_filename_target)

                            # Check if the file doesn't exist
                            if os.path.exists(feature_filename):
                                features = np.load(feature_filename)
                                labels = np.load(label_filename)

                                # Train and evaluate the different classifiers outlined in training.py
                                current_iteration += 1
                                print("Iteration " + str(current_iteration) + "/" + str(total_iterations))
                                run_res, run_param = test_classifier(features, labels, n_mfcc=n_mfcc, frame_size=frame_size,
                                                               n_segments=n_segments, models_used=models_used, test_size=test_size_val)
                                results.append(run_res)
                                parameters.append(run_param)
                            else:
                                print(
                                    "Error, data mismatch, features and labels data don't exist in features folder in test classifier")

    # After the loop you can convert results to a DataFrame and analyze it
    results_df = pd.DataFrame(results)
    parameters_df = pd.DataFrame(parameters)
    return results_df, parameters_df


# Balancing Dataset Function
def balance_dataset(features, labels, balance_method, oversampling_rate=0.6):
    # Check class distribution
    # Class 0: Negative Covid
    # Class 1: Positive Covid
    print("Before balancing:")
    class_0_sample_count = sum(labels == 0)
    class_1_sample_count = sum(labels == 1)
    total_original_samples = class_0_sample_count + class_1_sample_count
    print("Total Samples:", total_original_samples)
    print("Class 0:", class_0_sample_count)
    print("Class 1:", class_1_sample_count)

    match balance_method:
        case "Resampling":
            # Apply Random Over Sampling and Under Sampling
            # Define the resampling strategy
            over = RandomOverSampler(sampling_strategy=oversampling_rate)  # Oversample the minority to have 70% of the majority class
            under = RandomUnderSampler(sampling_strategy=1.0)  # Undersample the majority to have equal to the minority

            # First apply oversampling
            features_oversampled, labels_oversampled = over.fit_resample(features, labels)

            # print("After oversampling:")
            class_0_sample_count = sum(labels_oversampled == 0)
            class_1_sample_count = sum(labels_oversampled == 1)
            print("Total Samples:", class_0_sample_count + class_1_sample_count)
            print("Class 0:", class_0_sample_count)
            print("Class 1:", class_1_sample_count)

            # Then apply undersampling
            features_combined, labels_combined = under.fit_resample(features_oversampled, labels_oversampled)

            # Check new class distribution
            print("After undersampling:")
            class_0_sample_count = sum(labels_combined == 0)
            class_1_sample_count = sum(labels_combined == 1)
            total_final_samples = class_0_sample_count + class_1_sample_count
            print("Total Samples:", total_final_samples)
            print("Class 0:", class_0_sample_count)
            print("Class 1:", class_1_sample_count)

            return features_combined, labels_combined
        case "SMOTE":
            # Check class distribution before SMOTE
            # print("Before SMOTE:")
            class_0_sample_count = sum(labels == 0)
            class_1_sample_count = sum(labels == 1)
            total_original_samples = class_0_sample_count + class_1_sample_count
            print("Total Samples:", total_original_samples)
            print("Class 0:", class_0_sample_count)
            print("Class 1:", class_1_sample_count)

            # Apply SMOTE
            smote = SMOTE()  # random_state case
            # smote = SMOTE(random_state=42)
            features_res, labels_res = smote.fit_resample(features, labels)

            # Check class distribution after SMOTE
            # print("After SMOTE:")
            class_0_sample_count = sum(labels_res == 0)
            class_1_sample_count = sum(labels_res == 1)
            total_final_samples = class_0_sample_count + class_1_sample_count
            print("Total Samples:", total_final_samples)
            print("Class 0:", class_0_sample_count)
            print("Class 1:", class_1_sample_count)

            return features_res, labels_res
        case _:
            return False, False


# Test training function
def test_classifier(features, labels, n_mfcc=-1, frame_size=-1, n_segments=-1, models_used=None, test_size=0.2):
    if models_used is None:
        models_used = [0, 0, 0, 0, 0, 0]
        print("No model specified")

    shapes = [f.shape for f in features]
    unique_shapes = set(shapes)
    for shape in unique_shapes:
        print(f"Shape: {shape}, Count: {shapes.count(shape)}")

    # Balance dataset
    # Methods: Resampling, SMOTE
    balance_method = "SMOTE"

    # Test NaN conflict resolution
    # Methods: imputing, dropping
    test_nan_conflict_solving_method = "imputing"
    # features_combined, labels_combined = balance_dataset(features, labels, balance_method)

    # Test Ensemble Learning with majority class split
    # Method: 1 for normal learning, other odd values for ensemble learning. Most balanced ensemble value is 3
    dataset_splitting = 3

    # Split dataset
    # x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)  # random_state case
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, stratify=labels)
    print("x_train size: " + str(len(x_train)))

    # Check for NaN values in the data
    if np.any(np.isnan(x_train)):
        # print("NaN values found in x_train, applying imputation.")
        imputer = SimpleImputer(strategy='mean')
        x_train = imputer.fit_transform(x_train)

    if test_nan_conflict_solving_method == "imputing":
        # Check for NaN values in the scaled data
        if np.any(np.isnan(x_test)):
            # print("NaN values found in x_train, applying imputation.")
            imputer = SimpleImputer(strategy='mean')
            x_test = imputer.fit_transform(x_test)
        print("x_test size: " + str(len(x_test)))
    elif test_nan_conflict_solving_method == "dropping":
        # Drop samples with NaN values in test dataset
        print("x_test size before dropping: " + str(len(x_test)))

        # Create a mask for rows without NaN values
        mask = ~np.isnan(x_test).any(axis=1)
        if mask.any():
            # print("NaN values found in x_test, dropping samples.")
            pass

        # Apply the mask to x_test and y_test
        x_test = x_test[mask]
        y_test = y_test[mask]

        print("x_test size after dropping: " + str(len(x_test)))

    oversampling_rate = 0.6
    if dataset_splitting == 1:
        x_train_combined, y_train_combined = balance_dataset(x_train, y_train, balance_method, oversampling_rate)
    else:
        x_train_combined = x_train
        y_train_combined = y_train

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_combined)
    x_test = scaler.transform(x_test)

    # PCA to reduce the dimensionality of the features

    # Specify the number of principal components
    # n_components = 100  # Number of principal components to keep
    # pca = PCA(n_components=n_components)
    # reduced_x_train = pca.fit_transform(x_train)
    # reduced_x_test = pca.transform(x_test)

    # ALTERNATIVE: Specify the amount of variance to retain
    variance_ratio = 0.95  # Retain 95% of the variance
    pca = PCA(n_components=variance_ratio)
    reduced_x_train = pca.fit_transform(x_train)
    reduced_x_test = pca.transform(x_test)

    print("reduced_x_train: " + str(reduced_x_train.shape))
    print("reduced_x_test: " + str(reduced_x_test.shape))

    # Balance Test Dataset
    # Balance the test set using RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    reduced_x_test, y_test = rus.fit_resample(reduced_x_test, y_test)

    # Check Train Dataset Sample Distribution
    class_0_sample_count = sum(y_train_combined == 0)
    class_1_sample_count = sum(y_train_combined == 1)
    print("Total Samples Train:", class_0_sample_count + class_1_sample_count)
    print("Class 0:", class_0_sample_count)
    print("Class 1:", class_1_sample_count)

    # Check Test Dataset Sample Distribution
    class_0_sample_count = sum(y_test == 0)
    class_1_sample_count = sum(y_test == 1)
    print("Total Samples Test:", class_0_sample_count + class_1_sample_count)
    print("Class 0:", class_0_sample_count)
    print("Class 1:", class_1_sample_count)

    # TODO Investing the balance of test dataset
    # Initialize results

    parameters = {
        'train_samples_number': str(len(x_train)),
        'train_samples_reduced_number': str(len(reduced_x_train)),
        'test_samples_number': str(len(x_test)),
        'test_samples_reduced_number': str(len(reduced_x_test)),
        'test_size %': test_size * 50,  # TEMP FIX TO REPRESENT TEST DATASET CORRECTLY
        'ensemble learning groups': dataset_splitting,
        'dataset_balance_method': balance_method,
        'oversampling rate': oversampling_rate,
        'test_nan_conflict_solving_method': test_nan_conflict_solving_method,
        'mfcc': n_mfcc,
        'frame_size': frame_size,
        'segments': n_segments,
    }

    results_model = {}

    # Show iteration progress
    print("mfcc: " + str(n_mfcc))
    print("frame_size: " + str(frame_size))
    print("segments: " + str(n_segments))

    # Initialize all models
    # LR
    if models_used[0] == 1:
        # lr_hyper refers to the hyperparameters for lr
        # first slot is C, the regularization strength
        # Syntax: [a,b,c], Usage: 'C': np.logspace(a, b, c)

        lr_hyper = [[-7, 7, 15]]
        if dataset_splitting > 1:
            # Separate the majority and minority classes
            # Assuming you have features (X) and labels (y) as lists or arrays
            x = np.array(reduced_x_train)  # Convert to numpy array if they are not already
            y = np.array(y_train_combined)  # Convert to numpy array if they are not already

            # Separate the majority and minority classes
            minority_class_indices = np.where(y == 1)[0]  # Assuming minority class label is 1
            majority_class_indices = np.where(y == 0)[0]  # Assuming majority class label is 0

            x_minority = x[minority_class_indices]
            y_minority = y[minority_class_indices]

            x_majority = x[majority_class_indices]
            y_majority = y[majority_class_indices]

            # Shuffle the majority class
            # x_majority, y_majority = shuffle(x_majority, y_majority, random_state=42) # random state case
            x_majority, y_majority = shuffle(x_majority, y_majority)

            # Split the majority class into three parts
            split_size = len(x_majority) // 3

            x_majority_split_1 = x_majority[:split_size]
            y_majority_split_1 = y_majority[:split_size]

            x_majority_split_2 = x_majority[split_size:2 * split_size]
            y_majority_split_2 = y_majority[split_size:2 * split_size]

            x_majority_split_3 = x_majority[2 * split_size:]
            y_majority_split_3 = y_majority[2 * split_size:]

            # Create three balanced datasets
            x1 = np.concatenate((x_minority, x_majority_split_1), axis=0)
            y1 = np.concatenate((y_minority, y_majority_split_1), axis=0)

            x2 = np.concatenate((x_minority, x_majority_split_2), axis=0)
            y2 = np.concatenate((y_minority, y_majority_split_2), axis=0)

            x3 = np.concatenate((x_minority, x_majority_split_3), axis=0)
            y3 = np.concatenate((y_minority, y_majority_split_3), axis=0)

            # Shuffle the datasets to mix minority and majority instances
            # random state cases
            # x1, y1 = shuffle(x1, y1, random_state=42)
            # x2, y2 = shuffle(x2, y2, random_state=42)
            # x3, y3 = shuffle(x3, y3, random_state=42)
            x1, y1 = shuffle(x1, y1)
            x2, y2 = shuffle(x2, y2)
            x3, y3 = shuffle(x3, y3)

            print(len(x1))
            print(len(x2))
            print(len(x3))

            # Training
            model_lr_1, model_lr_hyper_1 = training.lr_training(x1, y1, lr_hyper)
            model_lr_2, model_lr_hyper_2 = training.lr_training(x2, y2, lr_hyper)
            model_lr_3, model_lr_hyper_3 = training.lr_training(x3, y3, lr_hyper)

            # Create a voting classifier
            ensemble = VotingClassifier(estimators=[('lr1', model_lr_1), ('lr2', model_lr_2), ('lr3', model_lr_3)], voting='soft')

            ensemble.fit(reduced_x_test, y_test)

            # Evaluating
            y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
            performance_metrics_lr = training.evaluate_model(y_test, y_pred_proba)

            # Saving the best hyperparameters
            results_model['Hyper_LR__C'] = model_lr_hyper_1["C"]
            results_model['performance_metrics_lr'] = performance_metrics_lr

        else:
            # Training
            model_lr, model_lr_hyper = training.lr_training(reduced_x_train, y_train_combined, lr_hyper)

            # Evaluating
            y_pred_proba = model_lr.predict_proba(reduced_x_test)[:, 1]
            performance_metrics_lr = training.evaluate_model(y_test, y_pred_proba)

            # Saving the best hyperparameters
            results_model['Hyper_LR__C'] = model_lr_hyper["C"]

            results_model['performance_metrics_lr'] = performance_metrics_lr

    if models_used[1] == 1:
        # knn_hyper refers to the hyperparameters for knn
        # first slot is number of neighbors
        # Syntax: [a,b,c], Usage: 'n_neighbors': list(range(a, b, c)),
        # second slot is leaf size
        # Syntax: [a,b,c], Usage: 'leaf_size': list(range(a, b, c)),

        knn_hyper = [[10, 101, 10], [5, 31, 5]]
        if dataset_splitting > 1:
            # Separate the majority and minority classes
            # Assuming you have features (X) and labels (y) as lists or arrays
            x = np.array(reduced_x_train)  # Convert to numpy array if they are not already
            y = np.array(y_train_combined)  # Convert to numpy array if they are not already

            # Separate the majority and minority classes
            minority_class_indices = np.where(y == 1)[0]  # Assuming minority class label is 1
            majority_class_indices = np.where(y == 0)[0]  # Assuming majority class label is 0

            x_minority = x[minority_class_indices]
            y_minority = y[minority_class_indices]

            x_majority = x[majority_class_indices]
            y_majority = y[majority_class_indices]

            # Shuffle the majority class
            # x_majority, y_majority = shuffle(x_majority, y_majority, random_state=42) # random state case
            x_majority, y_majority = shuffle(x_majority, y_majority)

            # Split the majority class into three parts
            split_size = len(x_majority) // 3

            x_majority_split_1 = x_majority[:split_size]
            y_majority_split_1 = y_majority[:split_size]

            x_majority_split_2 = x_majority[split_size:2 * split_size]
            y_majority_split_2 = y_majority[split_size:2 * split_size]

            x_majority_split_3 = x_majority[2 * split_size:]
            y_majority_split_3 = y_majority[2 * split_size:]

            # Create three balanced datasets
            x1 = np.concatenate((x_minority, x_majority_split_1), axis=0)
            y1 = np.concatenate((y_minority, y_majority_split_1), axis=0)

            x2 = np.concatenate((x_minority, x_majority_split_2), axis=0)
            y2 = np.concatenate((y_minority, y_majority_split_2), axis=0)

            x3 = np.concatenate((x_minority, x_majority_split_3), axis=0)
            y3 = np.concatenate((y_minority, y_majority_split_3), axis=0)

            # Shuffle the datasets to mix minority and majority instances
            # random state cases
            # x1, y1 = shuffle(x1, y1, random_state=42)
            # x2, y2 = shuffle(x2, y2, random_state=42)
            # x3, y3 = shuffle(x3, y3, random_state=42)
            x1, y1 = shuffle(x1, y1)
            x2, y2 = shuffle(x2, y2)
            x3, y3 = shuffle(x3, y3)

            print(len(x1))
            print(len(x2))
            print(len(x3))

            # Training
            model_knn_1, model_knn_hyper_1 = training.knn_training(x1, y1, knn_hyper)
            model_knn_2, model_knn_hyper_2 = training.knn_training(x2, y2, knn_hyper)
            model_knn_3, model_knn_hyper_3 = training.knn_training(x3, y3, knn_hyper)

            # Create a voting classifier
            ensemble = VotingClassifier(estimators=[('knn1', model_knn_1), ('knn2', model_knn_2), ('knn3', model_knn_3)], voting='soft')

            ensemble.fit(reduced_x_test, y_test)

            # Make predictions and evaluate the ensemble model
            y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
            performance_metrics_knn = training.evaluate_model(y_test, y_pred_proba)

            # Saving the best hyperparameters
            results_model['Hyper_kNN__n_neighbors'] = model_knn_hyper_1["n_neighbors"]
            results_model['Hyper_kNN__leaf_size'] = model_knn_hyper_1["leaf_size"]
            results_model['performance_metrics_knn'] = performance_metrics_knn
        else:
            # Training
            model_knn, model_knn_hyper = training.knn_training(reduced_x_train, y_train_combined, knn_hyper)

            # Evaluating
            y_pred_proba = model_knn.predict_proba(reduced_x_test)[:, 1]
            performance_metrics_knn = training.evaluate_model(y_test, y_pred_proba)

            # Saving the best hyperparameters
            results_model['Hyper_kNN__n_neighbors'] = model_knn_hyper["n_neighbors"]
            results_model['Hyper_kNN__leaf_size'] = model_knn_hyper["leaf_size"]

            results_model['performance_metrics_knn'] = performance_metrics_knn

    if models_used[2] == 1:
        # svm_hyper refers to the hyperparameters for svm
        # first slot is C, the regularization strength
        # Syntax: [a,b,c], Usage: 'C': np.logspace(a, b, c),
        # second slot is gamma, which controls the kernel coefficient
        # Syntax: [a,b,c], Usage: 'gamma': np.logspace(a, b, c)

        svm_hyper = [[-7, 7, 15], [-7, 7, 15]]
        # Training
        model_svm, model_svm_hyper = training.svm_training(reduced_x_train, y_train_combined, svm_hyper)

        # Evaluating
        y_pred_proba = model_svm.predict_proba(reduced_x_test)[:, 1]
        performance_metrics_svm = training.evaluate_model(y_test, y_pred_proba)

        # Saving the best hyperparameters
        results_model['Hyper_SVM__C'] = model_svm_hyper["C"]
        results_model['Hyper_SVM__gamma'] = model_svm_hyper["gamma"]

        results_model['performance_metrics_svm'] = performance_metrics_svm
    if models_used[3] == 1:
        # mlp_hyper refers to the hyperparameters for mlp
        # first slot is hidden_layer_sizes, the number of neurons
        # Syntax: [a,b,c], Usage: 'hidden_layer_sizes': [(n,) for n in range(a, b, c)],
        # second slot is alpha, which controls the L2 penalty
        # Syntax: [a,b], Usage: 'alpha': [10 ** i for i in range(a, b)]
        # third slot is learning_rate_init, which controls the stochastic gradient descent
        # Syntax: [a,b,c], Usage: 'learning_rate_init': np.arange(a, b, c)

        mlp_hyper = [[10, 101, 10], [-7, 8], [0.05, 1.05, 0.05]]
        # Training
        model_mlp, model_mlp_hyper = training.mlp_training(reduced_x_train, y_train_combined, mlp_hyper)

        # Evaluating
        y_pred_proba = model_mlp.predict_proba(reduced_x_test)[:, 1]
        performance_metrics_mlp = training.evaluate_model(y_test, y_pred_proba)

        # Saving the best hyperparameters
        results_model['Hyper_MLP__hidden_layer_sizes'] = model_mlp_hyper["hidden_layer_sizes"]
        results_model['Hyper_MLP__alpha'] = model_mlp_hyper["alpha"]
        results_model['Hyper_MLP__learning_rate_init'] = model_mlp_hyper["learning_rate_init"]

        results_model['performance_metrics_mlp'] = performance_metrics_mlp
    if models_used[4] == 1:
        # cnn_hyper refers to the hyperparameters for cnn
        # first slot is num_filters, the No. of Conv filters
        # Syntax: [a,b,c], Usage: 'num_filters': [3 * 2 ** k4 for k4 in range(a, b)],
        # second slot is kernel_size, which controls the Kernel size
        # Syntax: [a,b], Usage: 'kernel_size': range(a, b)
        # third slot is dropout_rate, which controls the Dropout rate
        # Syntax: [a,b,c], Usage: 'dropout_rate': range(a, b, c)
        # fourth slot is dense_size, the Dense layer size
        # Syntax: [a,b], Usage: 'dense_size': [2 ** k5 for k5 in range(a, b)],
        # fifth slot is batch_size, which controls the Batch Size
        # Syntax: [a,b], Usage: 'batch_size': [2 ** k8 for k8 in range(a, b)]
        # sixth slot is epochs, which controls the No. of epochs
        # Syntax: [a,b,c], Usage: 'epochs': range(a, b, c)

        cnn_hyper = [[3, 6], [2, 4], [0.1, 0.6, 0.2], [4, 6], [6, 9], [10, 260, 20]]
        # Training
        model_cnn, model_cnn_hyper = training.cnn_training(reduced_x_train, y_train_combined, cnn_hyper)

        # Evaluating
        y_pred_proba = model_cnn.predict_proba(reduced_x_test)[:, 1]
        performance_metrics_cnn = training.evaluate_model(y_test, y_pred_proba)

        # Saving the best hyperparameters
        results_model['Hyper_CNN__num_filters'] = model_cnn_hyper["num_filters"]
        results_model['Hyper_CNN__kernel_size'] = model_cnn_hyper["kernel_size"]
        results_model['Hyper_CNN__dropout_rate'] = model_cnn_hyper["dropout_rate"]
        results_model['Hyper_CNN__dense_size'] = model_cnn_hyper["dense_size"]
        results_model['Hyper_CNN__batch_size'] = model_cnn_hyper["batch_size"]
        results_model['Hyper_CNN__epochs'] = model_cnn_hyper["epochs"]

        results_model['performance_metrics_cnn'] = performance_metrics_cnn
    if models_used[6] == 1:
        lr_hyper = [[-7, 7, 15]]
        if dataset_splitting > 1:
            # Separate the majority and minority classes
            # Assuming you have features (X) and labels (y) as lists or arrays
            x = np.array(reduced_x_train)  # Convert to numpy array if they are not already
            y = np.array(y_train_combined)  # Convert to numpy array if they are not already

            # Separate the majority and minority classes
            minority_class_indices = np.where(y == 1)[0]  # Assuming minority class label is 1
            majority_class_indices = np.where(y == 0)[0]  # Assuming majority class label is 0

            x_minority = x[minority_class_indices]
            y_minority = y[minority_class_indices]

            x_majority = x[majority_class_indices]
            y_majority = y[majority_class_indices]

            # Shuffle the majority class
            # x_majority, y_majority = shuffle(x_majority, y_majority, random_state=42) # random state case
            x_majority, y_majority = shuffle(x_majority, y_majority)

            # Split the majority class into three parts
            split_size = len(x_majority) // 3

            x_majority_split_1 = x_majority[:split_size]
            y_majority_split_1 = y_majority[:split_size]

            x_majority_split_2 = x_majority[split_size:2 * split_size]
            y_majority_split_2 = y_majority[split_size:2 * split_size]

            x_majority_split_3 = x_majority[2 * split_size:]
            y_majority_split_3 = y_majority[2 * split_size:]

            # Create three balanced datasets
            x1 = np.concatenate((x_minority, x_majority_split_1), axis=0)
            y1 = np.concatenate((y_minority, y_majority_split_1), axis=0)

            x2 = np.concatenate((x_minority, x_majority_split_2), axis=0)
            y2 = np.concatenate((y_minority, y_majority_split_2), axis=0)

            x3 = np.concatenate((x_minority, x_majority_split_3), axis=0)
            y3 = np.concatenate((y_minority, y_majority_split_3), axis=0)

            # Shuffle the datasets to mix minority and majority instances
            # random state cases
            # x1, y1 = shuffle(x1, y1, random_state=42)
            # x2, y2 = shuffle(x2, y2, random_state=42)
            # x3, y3 = shuffle(x3, y3, random_state=42)
            x1, y1 = shuffle(x1, y1)
            x2, y2 = shuffle(x2, y2)
            x3, y3 = shuffle(x3, y3)

            print(len(x1))
            print(len(x2))
            print(len(x3))

            # Training
            model_lr_1, model_lr_hyper_1 = training.lr_training(x1, y1, lr_hyper)
            model_lr_2, model_lr_hyper_2 = training.lr_training(x2, y2, lr_hyper)
            model_lr_3, model_lr_hyper_3 = training.lr_training(x3, y3, lr_hyper)

            # Create a voting classifier
            # ensemble = VotingClassifier(estimators=[('lr1', model_lr_1), ('lr2', model_lr_2), ('lr3', model_lr_3)], voting='soft')

            # ensemble.fit(reduced_x_test, y_test)

            # Evaluating
            # y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
            # performance_metrics_lr = training.evaluate_model(y_test, y_pred_proba)

            # Saving the best hyperparameters
            results_model['Hyper_LR__C'] = model_lr_hyper_1["C"]
            # results_model['performance_metrics_lr'] = performance_metrics_lr

            # TEMP
            knn_hyper = [[10, 101, 10], [5, 31, 5]]
            # Training
            model_knn_1, model_knn_hyper_1 = training.knn_training(x1, y1, knn_hyper)
            model_knn_2, model_knn_hyper_2 = training.knn_training(x2, y2, knn_hyper)
            model_knn_3, model_knn_hyper_3 = training.knn_training(x3, y3, knn_hyper)

            # Create a voting classifier
            ensemble = VotingClassifier(
                estimators=[('lr1', model_lr_1), ('lr2', model_lr_2), ('lr3', model_lr_3), ('knn1', model_knn_1),
                            ('knn2', model_knn_2), ('knn3', model_knn_3)], voting='soft')

            ensemble.fit(reduced_x_test, y_test)

            # Make predictions and evaluate the ensemble model
            y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
            performance_metrics_test = training.evaluate_model(y_test, y_pred_proba)

            # Saving the best hyperparameters
            results_model['Hyper_kNN__n_neighbors'] = model_knn_hyper_1["n_neighbors"]
            results_model['Hyper_kNN__leaf_size'] = model_knn_hyper_1["leaf_size"]
            # results_model['performance_metrics_knn'] = performance_metrics_knn

            results_model['performance_metrics_test'] = performance_metrics_test

    return results_model, parameters


# Convert models_used values to their names
def models_name_conv(model):
    switcher = {
        0: "performance_metrics_lr",
        1: "performance_metrics_knn",
        2: "performance_metrics_svm",
        3: "performance_metrics_mlp",
        4: "performance_metrics_cnn",
        5: "performance_metrics_lstm",
        6: "performance_metrics_test"
    }

    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return switcher.get(model, "Model Name Mismatch")


# Function to display the results dataframe in a better way
def test_display(results_df, models_used_str, parameters_df):
    # Create a new DataFrame to store the results
    # expanded_results_df = pd.DataFrame()

    # Print the entire DataFrame
    print("Metrics Used: [specificity, sensitivity, precision, accuracy, F1, AUC]")
    # Get the current date
    current_date = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    for perf_res in models_used_str:
        if perf_res in results_df:
            # print(perf_res)
            metrics = results_df[perf_res]
            print(metrics.apply(lambda x: [f"{num:.4f}" for num in x]))

            # Convert the 'array_column' to a DataFrame and expand it into separate columns
            array_df = pd.DataFrame(metrics.tolist(), index=results_df.index)

            # Your specific list of names for the expanded columns
            column_names = [perf_res + "_specificity", perf_res + "_sensitivity", perf_res + "_precision",
                            perf_res + "_accuracy", perf_res + "_F1", perf_res + "_AUC"]

            # Ensure the list length matches the number of columns to rename
            if len(column_names) == array_df.shape[1]:
                array_df.columns = column_names
            else:
                raise ValueError("The number of column names does not match the number of columns.")

            # Join the new columns with the expanded_results_df DataFrame
            parameters_df = pd.concat([parameters_df, array_df], axis=1)

            # print(expanded_results_df)

            metrics_folder = "./" + data_dir_choice + "/model_metrics"

            # Check if the directory exists, if not, create it
            if not os.path.exists(metrics_folder):
                os.makedirs(metrics_folder)

            # Find the model with the best ROC curve for each model type used
            max_row = parameters_df.loc[parameters_df[perf_res + "_AUC"].idxmax()]
            print(max_row)

            # Create a blank row to separate the best results
            first_blank_row = pd.DataFrame([{}], columns=parameters_df.columns)

            # Create a second blank row to separate the best results
            # Adding the max row label of each model used
            second_blank_row = pd.DataFrame({parameters_df.columns[0]: ["Best Model for " + perf_res]}, index=[0])
            second_blank_row = second_blank_row.reindex(columns=parameters_df.columns, fill_value='')

            # Append the blank rows followed by the max row to the DataFrame
            parameters_df = pd.concat([parameters_df, first_blank_row, second_blank_row, pd.DataFrame([max_row])],
                                            ignore_index=True)

            # Save the expanded DataFrame to a CSV file
            parameters_df.to_csv('./' + data_dir_choice + '/model_metrics/my_dataframe_expanded_' + current_date + '.csv',
                                       index=False)


results_df = test_modular()
