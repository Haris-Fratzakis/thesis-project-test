import os
import random
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode, ttest_ind
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
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
data_dir_choice = "smarty4covid"

# IMPORTANT NOTICE
# If run on different PCs, this needs to be changed to point to the dataset directory
# Every dataset path query is formed in relation to this variable (data_dir)
# Load the index csv
if data_dir_choice == "smarty4covid":
    data_dir = "E:/Storage/University/Thesis/smarty4covid/"
elif data_dir_choice == "coswara":
    data_dir = "E:/Storage/University/Thesis/iiscleap-Coswara-Data-bf300ae/Extracted_data_mix"
else:
    print("No Data Specified, thus defaulting to smarty4covid")
    data_dir = "E:/Storage/University/Thesis/smarty4covid/"

# Setting a global random value to use in random states in the rest of the code
# So multiple successive iterations of the program can work on the same data in the same way
# in order for proper comparisons to be made
# random_state_global_value = random.randint(1, 1000)
random_state_global_value = 227


# This function gets called first
# Function to have modular feature extraction and training methods
def modular_model_training():
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
        print("Incorrect dataset specified, defaulting to smarty4covid")
        data_index = os.path.join(data_dir, 'smarty4covid_tabular_data.csv')
        data = pd.read_csv(data_index)

        # Exclude rows where 'covid_status' is 'no'
        data = data[data.covid_status != 'no']

    # Modular feature extraction stage

    # Hyperparameter values based on:
    # Preliminary diagnosis of COVID-19 based on cough sounds using machine learning algorithms
    # https://ieeexplore.ieee.org/abstract/document/9432324
    # Full list of parameters below
    # k_values_mfcc = [1, 2, 3, 4, 5]
    # k_values_frame = [8, 9, 10, 11, 12]
    # k_values_segment = [5, 7, 10, 12, 15]

    k_values_mfcc = [5]
    k_values_frame = [8, 9, 10, 11, 12]
    k_values_segment = [5, 7, 10, 12, 15]

    modular_feat_extr(data=data, k_values_mfcc=k_values_mfcc, k_values_frame=k_values_frame,
                      k_values_segment=k_values_segment)

    # models_used signifies which model is used, each slot signifies a different model
    # 1 means model is going to be used, 0 means it will not be used
    # slots are [LR, KNN, SVM, MLP, CNN, LSTM]
    models_used = [1, 0, 0, 0, 0, 0]
    test_size = [0.2]
    # This is the modular classifier training stage
    results_df, parameters_df = modular_classifier(k_values_mfcc=k_values_mfcc, k_values_frame=k_values_frame,
                                                   k_values_segment=k_values_segment, models_used=models_used,
                                                   test_size=test_size)

    models_used = models_used_name_converter(models_used)
    test_display(results_df, models_used, parameters_df)

    return results_df


# Function that converts models_used values to their names
def models_used_name_converter(models_used):
    models_used_temp = [0, 0, 0, 0, 0, 0]

    for idx in range(len(models_used)):
        if models_used[idx] == 1:
            models_used_temp[idx] = models_name_switch_table(idx)

    print("Models Used List: " + str(models_used_temp))
    return models_used_temp


# Function that makes sure all features vectors are the same shape
def pad_or_truncate(features, target_length):
    if len(features) > target_length:
        return features[:target_length]
    elif len(features) < target_length:
        return np.pad(features, (0, target_length - len(features)), 'constant')
    else:
        return features


# Function for feature extraction
def modular_feat_extr(data, k_values_mfcc=None, k_values_frame=None, k_values_segment=None):
    if k_values_mfcc is None:
        k_values_mfcc = [1]
    if k_values_frame is None:
        k_values_frame = [-1]
    if k_values_segment is None:
        k_values_segment = [-1]

    # Initialize the LabelEncoder
    le = LabelEncoder()

    # Calculate the amount of iterations that feature extraction has to go through
    if k_values_frame == [-1] or k_values_segment == [-1]:
        total_iterations = 0
    else:
        total_iterations = len(k_values_mfcc) * len(k_values_frame) * len(k_values_segment)
    current_iteration = 0

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

            # Create the filename
            feature_filename_target = "extracted_features_" + str(k_mfcc) + ".npy"
            label_filename_target = "extracted_labels_" + str(k_mfcc) + ".npy"
            feature_filename = os.path.join(features_folder, feature_filename_target)
            label_filename = os.path.join(features_folder, label_filename_target)

            successful_indices = []
            features_list = []

            # Check if the file doesn't exist
            if not os.path.exists(feature_filename):
                # Extract features and simultaneously filter labels
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
                        # Extract features and simultaneously filter labels
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
                            feat = feat_extr.extract_features(data_dir, audio_path, audio_name, n_mfcc, frame_size, hop_length)
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

                        current_iteration += 1
                        print("Feature Extraction Iteration " + str(current_iteration) + "/" + str(total_iterations))

                        # Feature Extraction Initialization Function with all five methods and all three hyperparameters

                        # Name of the directory and file where the features will be saved
                        features_folder = data_dir_choice + "/extracted_features/feat_extr_with_segm"

                        # Check if the directory exists, if not, create it
                        if not os.path.exists(features_folder):
                            os.makedirs(features_folder)

                        feature_filename_target = "extracted_features_" + str(k_mfcc) + "_" + str(k_frame) + "_" + str(k_segment) + ".npy"
                        label_filename_target = "extracted_labels_" + str(k_mfcc) + "_" + str(k_frame) + "_" + str(k_segment) + ".npy"
                        feature_filename = os.path.join(features_folder, feature_filename_target)
                        label_filename = os.path.join(features_folder, label_filename_target)

                        successful_indices = []
                        features_list = []

                        # Check if the file doesn't exist
                        if not os.path.exists(feature_filename):
                            # Extract features and simultaneously filter labels
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
                                feat = feat_extr.extract_features_with_segments(data_dir, audio_path, audio_name, n_mfcc, frame_size, hop_length, n_segments)
                                if feat is not False:
                                    features_list.append(feat)
                                    successful_indices.append(idx)

                            # Check if all features have the same shape
                            shapes = [f.shape for f in features_list]
                            unique_shapes = set(shapes)
                            for shape in unique_shapes:
                                print(f"Shape: {shape}, Count: {shapes.count(shape)}")
                            if len(unique_shapes) > 1:
                                print("Inconsistent shapes found in features_list:")
                                for shape in unique_shapes:
                                    print(f"Shape: {shape}, Count: {shapes.count(shape)}")

                                lengths = [len(f) for f in features_list]

                                # Check if there are different size vectors
                                most_common_length_result = mode(lengths)
                                if isinstance(most_common_length_result.mode, np.ndarray):
                                    most_common_length = most_common_length_result.mode[0]
                                else:
                                    most_common_length = most_common_length_result.mode

                                # Shape all features vectors to use the most common length
                                target_length = most_common_length
                                features_list = [pad_or_truncate(f, target_length) for f in features_list]
                            features = np.array(features_list)

                            # Filter labels based on successful feature extraction
                            labels = np.array(data.loc[successful_indices, 'covid_status'])

                            # Convert labels to a consistent numerical format
                            labels = le.fit_transform(labels)

                            np.save(feature_filename, features)
                            np.save(label_filename, labels)


# Modular classifier function
def modular_classifier(k_values_mfcc, k_values_frame=None, k_values_segment=None, models_used=None, test_size=None):
    if k_values_mfcc is None:
        k_values_mfcc = [1]
    if k_values_frame is None:
        k_values_frame = [-1]
    if k_values_segment is None:
        k_values_segment = [-1]
    if models_used is None:
        models_used = [0, 0, 0, 0, 0, 0]
    if test_size is None:
        k_values_segment = [0.2]

    # Initialize a list for storing the results
    results = []
    parameters = []

    total_iterations = len(k_values_mfcc) * len(k_values_frame) * len(k_values_segment) * len(test_size)
    current_iteration = 0
    iteration_identifier = random_state_global_value
    # Loop over all combinations of hyperparameters
    for test_size_val in test_size:
        for k_mfcc in k_values_mfcc:
            n_mfcc = 14 * k_mfcc

            # Feature Extraction Initialization Function with only one method and only one hyperparameter
            if k_values_frame == [-1]:
                # Name of the directory and file where the features will be saved
                features_folder = data_dir_choice + "/extracted_features/feat_extr_simple"

                # Check if the directory exists, if not, report the error
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
                    save_iteration_csv(pd.DataFrame(results), models_used_name_converter(models_used), pd.DataFrame(parameters), iteration_identifier)
                else:
                    print("Error, data mismatch, features and labels data don't exist in features folder in test classifier")
            else:
                for k_frame in k_values_frame:
                    frame_size = 2 ** k_frame

                    # Feature Extraction Initialization Function with all five methods and only two hyperparameters
                    if k_values_segment == [-1]:
                        # Name of the directory and file where the features will be saved
                        features_folder = data_dir_choice + "/extracted_features/feat_extr"

                        # Check if the directory exists, if not, report the error
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
                            run_res, run_param = test_classifier(features, labels, n_mfcc=n_mfcc, frame_size=frame_size, models_used=models_used, test_size=test_size_val)
                            results.append(run_res)
                            parameters.append(run_param)
                            save_iteration_csv(pd.DataFrame(results), models_used_name_converter(models_used), pd.DataFrame(parameters), iteration_identifier)
                        else:
                            print("Error, data mismatch, features and labels data don't exist in features folder in test classifier")
                    else:
                        for k_segment in k_values_segment:
                            n_segments = 10 * k_segment

                            # Feature Extraction Initialization Function with all five methods and all three hyperparameters
                            # Name of the directory and file where the features will be saved
                            features_folder = data_dir_choice + "/extracted_features/feat_extr_with_segm"

                            # Check if the directory exists, if not, report the error
                            if not os.path.exists(features_folder):
                                print("Error, data mismatch, features folder doesn't exist in test classifier")

                            feature_filename_target = "extracted_features_" + str(k_mfcc) + "_" + str(
                                k_frame) + "_" + str(
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
                                run_res, run_param = test_classifier(features, labels, n_mfcc=n_mfcc, frame_size=frame_size, n_segments=n_segments, models_used=models_used, test_size=test_size_val)
                                results.append(run_res)
                                parameters.append(run_param)
                                save_iteration_csv(pd.DataFrame(results), models_used_name_converter(models_used), pd.DataFrame(parameters), iteration_identifier)
                            else:
                                print("Error, data mismatch, features and labels data don't exist in features folder in test classifier")

    # After the loop you can convert results to a DataFrame and analyze it
    results_df = pd.DataFrame(results)
    parameters_df = pd.DataFrame(parameters)
    return results_df, parameters_df


# Function for balancing the dataset
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

            # With ensemble training method, oversampling is not necessary
            # # Define the resampling strategy
            # over = RandomOverSampler(sampling_strategy=oversampling_rate, random_state=random_state_global_value)  # Oversample the minority to have oversampling_rate * 100 % of the majority class
            #
            # # Apply oversampling
            # features_oversampled, labels_oversampled = over.fit_resample(features, labels)
            #
            # print("After oversampling:")
            # class_0_sample_count = sum(labels_oversampled == 0)
            # class_1_sample_count = sum(labels_oversampled == 1)
            # print("Total Samples:", class_0_sample_count + class_1_sample_count)
            # print("Class 0:", class_0_sample_count)
            # print("Class 1:", class_1_sample_count)

            # Apply undersampling

            under = RandomUnderSampler(sampling_strategy=1.0, random_state=random_state_global_value)  # Undersample the majority to have equal to the minority

            # features_combined, labels_combined = under.fit_resample(features_oversampled, labels_oversampled)
            features_combined, labels_combined = under.fit_resample(features, labels)

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
            print("Before SMOTE:")
            class_0_sample_count = sum(labels == 0)
            class_1_sample_count = sum(labels == 1)
            total_original_samples = class_0_sample_count + class_1_sample_count
            print("Total Samples:", total_original_samples)
            print("Class 0:", class_0_sample_count)
            print("Class 1:", class_1_sample_count)

            # Apply SMOTE
            smote = SMOTE(random_state=random_state_global_value)
            features_res, labels_res = smote.fit_resample(features, labels)

            # Check class distribution after SMOTE
            print("After SMOTE:")
            class_0_sample_count = sum(labels_res == 0)
            class_1_sample_count = sum(labels_res == 1)
            total_final_samples = class_0_sample_count + class_1_sample_count
            print("Total Samples:", total_final_samples)
            print("Class 0:", class_0_sample_count)
            print("Class 1:", class_1_sample_count)

            return features_res, labels_res
        case _:
            return False, False


# Function to split training datasets for ensemble training
def training_ensemble_split(reduced_x_train, y_train_combined, dataset_splitting):
    # Random_state case
    random_state = random_state_global_value
    if random_state is not None:
        np.random.seed(random_state)

    # Separate the majority and minority classes
    x = np.array(reduced_x_train)  # Convert to numpy array if they are not already
    y = np.array(y_train_combined)  # Convert to numpy array if they are not already

    # Separate the majority and minority classes
    minority_class_indices = np.where(y == 1)[0]
    majority_class_indices = np.where(y == 0)[0]

    x_minority = x[minority_class_indices]
    y_minority = y[minority_class_indices]

    x_majority = x[majority_class_indices]
    y_majority = y[majority_class_indices]

    # TODO All shuffles are temporarily disabled to investigate the split
    # Shuffle the majority class
    # x_majority, y_majority = shuffle(x_majority, y_majority, random_state=random_state)

    # Split the majority class into three parts
    split_size = len(x_majority) // dataset_splitting

    x_majority_split = []
    y_majority_split = []

    x_split_dataset = []
    y_split_dataset = []

    for i in range(dataset_splitting - 1):
        x_majority_split.append(x_majority[i * split_size: (i + 1) * split_size])
        y_majority_split.append(y_majority[i * split_size: (i + 1) * split_size])

        # Create balanced datasets
        x_split_dataset.append(np.concatenate((x_minority, x_majority_split[i]), axis=0))
        y_split_dataset.append(np.concatenate((y_minority, y_majority_split[i]), axis=0))

        # Shuffle the datasets to mix minority and majority instances
        # x_split_dataset[i], y_split_dataset[i] = shuffle(x_split_dataset[i], y_split_dataset[i], random_state=random_state)

        print(len(x_split_dataset[i]))

    # Create last ensemble dataset too
    x_majority_split.append(x_majority[(dataset_splitting - 1) * split_size:])
    y_majority_split.append(y_majority[(dataset_splitting - 1) * split_size:])

    x_split_dataset.append(np.concatenate((x_minority, x_majority_split[dataset_splitting - 1]), axis=0))
    y_split_dataset.append(np.concatenate((y_minority, y_majority_split[dataset_splitting - 1]), axis=0))

    # Shuffle the last ensemble dataset
    # x_split_dataset[dataset_splitting - 1], y_split_dataset[dataset_splitting - 1] = shuffle(x_split_dataset[dataset_splitting - 1], y_split_dataset[dataset_splitting - 1], random_state=random_state)

    print(len(x_split_dataset[dataset_splitting - 1]))

    return x_split_dataset, y_split_dataset


# Custom Scaler Function
def custom_scaler(features, scaler_type):
    if scaler_type == "standard":
        # Standard scaler
        num_samples, num_features = features.shape
        print("features.shape: ", features.shape)

        # Make an Array with num_samples empty arrays
        standardized_data = []
        for _ in range(num_samples):
            standardized_data.append([])

        # features = [[sample 1 feature 1, sample 1 feature 2, etc], [sample 2 feature 1, sample 2 feature 2, etc], etc]
        for i in range(num_features):
            temp_features_total = []
            temp__standardized_features_total = []
            for j in range(num_samples):
                temp_features_total.append(features[j][i])
            # temp_features_total = [sample 1 feature 1, sample 2 feature 1, etc]
            mean = np.mean(temp_features_total)
            std_dev = np.std(temp_features_total)
            for j in range(num_samples):
                standardized_data[j].append((features[j][i] - mean) / std_dev)
            # standardized_data = [[sample 1 standardized feature 1, sample 1 standardized feature 2, etc], [sample 2
            # standardized feature 1, sample 2 standardized feature 2, etc], etc]

            for j in range(num_samples):
                temp__standardized_features_total.append(standardized_data[j][i])
            # print("Standardized mean: ", np.mean(temp__standardized_features_total))
            # print("Standardized std_dev: ", np.std(temp__standardized_features_total))

        np_standardized_data = np.array(standardized_data)
        print("standardized_data.shape: ", np_standardized_data.shape)
        return np_standardized_data
    elif scaler_type == "min_max":
        # Min max scaler with range [0,1]
        num_samples, num_features = features.shape
        print("features.shape: ", features.shape)

        # Make an Array with num_samples empty arrays
        min_max_data = []
        for _ in range(num_samples):
            min_max_data.append([])

        # features = [[sample 1 feature 1, sample 1 feature 2, etc], [sample 2 feature 1, sample 2 feature 2, etc], etc]
        for i in range(num_features):
            # print("features: ", features)
            temp_features_total = []
            temp_min_max_features_total = []
            for j in range(num_samples):
                temp_features_total.append(features[j][i])
            # print("temp_features_total[:100]", temp_features_total[:100])
            # temp_features_total = [sample 1 feature 1, sample 2 feature 1, etc]
            feature_min = np.min(temp_features_total)
            feature_max = np.max(temp_features_total)
            # print("feature_min: ", feature_min)
            # print("feature_max: ", feature_max)
            for j in range(num_samples):
                min_max_data[j].append((features[j][i] - feature_min) / (feature_max - feature_min))
            # min_max_data = [[sample 1 min_max feature 1, sample 1 min_max feature 2, etc], [sample 2 min_max feature 1, sample 2 min_max feature 2, etc], etc]

            for j in range(num_samples):
                temp_min_max_features_total.append(min_max_data[j][i])
            # temp_min_max_features_total = [sample 1 feature 1, sample 2 feature 1, etc]
            # print("Min_max mean: ", np.mean(temp_min_max_features_total))
            # print("Min_max std_dev: ", np.std(temp_min_max_features_total))

        np_min_max_data = np.array(min_max_data)
        print("min_max_data.shape: ", np_min_max_data.shape)
        return np_min_max_data
    else:
        # Default Standard scaler
        print("Wrong scaler_type, defaulting to Standard Scaler")
        num_samples, num_features = features.shape
        print("features.shape: ", features.shape)

        # Make an Array with num_samples empty arrays
        standardized_data = []
        for _ in range(num_samples):
            standardized_data.append([])

        # features = [[sample 1 feature 1, sample 1 feature 2, etc], [sample 2 feature 1, sample 2 feature 2, etc], etc]
        for i in range(num_features):
            temp_features_total = []
            temp__standardized_features_total = []
            for j in range(num_samples):
                temp_features_total.append(features[j][i])
            # temp_features_total = [sample 1 feature 1, sample 2 feature 1, etc]
            mean = np.mean(temp_features_total)
            std_dev = np.std(temp_features_total)
            for j in range(num_samples):
                standardized_data[j].append((features[j][i] - mean) / std_dev)
            # standardized_data = [[sample 1 standardized feature 1, sample 1 standardized feature 2, etc], [sample 2
            # standardized feature 1, sample 2 standardized feature 2, etc], etc]

            for j in range(num_samples):
                temp__standardized_features_total.append(standardized_data[j][i])
            # print("Standardized mean: ", np.mean(temp__standardized_features_total))
            # print("Standardized std_dev: ", np.std(temp__standardized_features_total))

        np_standardized_data = np.array(standardized_data)
        print("standardized_data.shape: ", np_standardized_data.shape)
        return np_standardized_data


# Test training function
def test_classifier(features, labels, n_mfcc=-1, frame_size=-1, n_segments=-1, models_used=None, test_size=0.2):
    if models_used is None:
        models_used = [0, 0, 0, 0, 0, 0]
        print("No model specified")

    # Random_state case
    random_state = random_state_global_value
    if random_state is not None:
        np.random.seed(random_state)

    shapes = [f.shape for f in features]
    unique_shapes = set(shapes)
    for shape in unique_shapes:
        print(f"Shape: {shape}, Count: {shapes.count(shape)}")

    # Balance dataset
    # Methods: Resampling, SMOTE
    balance_method = "Resampling"

    # Test NaN conflict resolution
    # Methods: imputing, dropping
    test_nan_conflict_solving_method = "dropping"
    # features_combined, labels_combined = balance_dataset(features, labels, balance_method)

    # Test Ensemble Learning with majority class split
    # Method: 1 for normal learning, other odd values for ensemble learning
    # Most balanced ensemble value was 3 with the old method
    # Most balanced ensemble value is 5 with the new method
    dataset_splitting = 5

    # Method for splitting the dataset into train and test
    # old_method: The built-in method from sklearn (train_test_split), which leads to an imbalanced test dataset
    # new_method: My custom function to keep test dataset balanced between classes
    train_test_split_method = "new_method"

    # Drop samples with NaN values in the dataset
    print("Dataset before dropping: " + str(len(features)))
    print("features[0]]: ", features[0][:10])
    print("features[1]]: ", features[1][:10])
    print("features[2]]: ", features[2][:10])

    # Create a mask for rows without NaN values
    mask = ~np.isnan(features).any(axis=1)
    if mask.any():
        # print("NaN values found in x_test, dropping samples.")
        pass

    # Apply the mask to x_test and y_test
    features_cleaned = features[mask]
    labels_cleaned = labels[mask]

    print("Dataset after dropping: " + str(len(features_cleaned)))
    print("features_cleaned[0]]: ", features_cleaned[0][:10])
    print("features_cleaned[1]]: ", features_cleaned[1][:10])
    print("features_cleaned[2]]: ", features_cleaned[2][:10])

    # Standardize features
    # scaler = StandardScaler()
    # features_scaled = scaler.fit_transform(features_cleaned)

    # Custom Scaler
    # scaler_type choice
    # Methods: "standard", "min_max"
    scaler_type = "min_max"
    features_scaled = custom_scaler(features_cleaned, scaler_type)

    print("Scaling Dataset Complete")
    print("features_scaled[0]]: ", features_scaled[0][:10])
    print("features_scaled[1]]: ", features_scaled[1][:10])
    print("features_scaled[2]]: ", features_scaled[2][:10])

    # # Test plot
    # matplotlib.use('Agg')  # Use a non-interactive backend like 'Agg' for script execution
    # features_original = []
    # features_standardized = []
    # # Create x-axis labels for all features across all samples
    # num_samples, num_features = features_cleaned.shape
    # print("num_samples", num_samples)
    # print("num_features", num_features)
    # for i in range(num_features):
    #     temp_original = []
    #     temp_standardized = []
    #     for j in range(num_samples):
    #         temp_original.append(features_cleaned[j][i])
    #         temp_standardized.append(features_scaled[j][i])
    #     features_original.append(temp_original)
    #     features_standardized.append(temp_standardized)
    #
    # # Plotting
    # fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    #
    # print("calc done")
    #
    # # Plot flattened original data
    # axes[0].plot(features_original, marker='o')
    # axes[0].set_title('Original Data')
    # axes[0].set_xlabel('Features for All Samples')
    # axes[0].set_ylabel('Value')
    # # axes[0].legend()
    #
    # print("original data done")
    #
    # # Plot flattened standardized data
    # axes[1].plot(features_standardized, marker='o')
    # axes[1].set_title('Standardized Data')
    # axes[1].set_xlabel('Features for All Samples')
    # axes[1].set_ylabel('Value')
    # # axes[1].legend()
    #
    # print("scaled data done")
    #
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('test.png')
    #
    # print("plot done")

    # Split dataset
    if train_test_split_method == "new_method":

        # Separate the data by class
        class_0_indices = np.where(labels_cleaned == 0)[0]
        print("class_0_indices: ", len(class_0_indices))
        class_1_indices = np.where(labels_cleaned == 1)[0]
        print("class_1_indices: ", len(class_1_indices))

        # Calculate the number of samples to be included in the test set for each class
        n_samples = int(len(labels_cleaned) * test_size // 2)

        # Randomly sample indices for the test set
        test_class_0_indices = np.random.choice(class_0_indices, size=n_samples, replace=False)
        print("test_class_0_indices: ", len(test_class_0_indices))
        test_class_1_indices = np.random.choice(class_1_indices, size=n_samples, replace=False)
        print("test_class_1_indices: ", len(test_class_1_indices))

        # Combine test indices
        test_indices = np.concatenate([test_class_0_indices, test_class_1_indices])
        # print("test_indices: ", np.sort(test_indices))
        print("test_indices len: ", len(test_indices))

        # Create the test set
        x_test = features_scaled[test_indices]
        y_test = labels_cleaned[test_indices]
        # print("x_test[0]: ", x_test[0][:100])
        # print("x_test[1]: ", x_test[1][:100])
        # print("x_test[2]: ", x_test[2][:100])

        # Create the train set by excluding the test set indices
        train_indices = np.setdiff1d(np.arange(len(labels_cleaned)), test_indices)
        # print("train_indices: ", train_indices[:20])
        print("train_indices len: ", len(train_indices))

        # Shuffle the train indices
        # np.random.shuffle(train_indices)

        # Index the training set
        x_train = features_scaled[train_indices]
        y_train = labels_cleaned[train_indices]
    else:
        x_train, x_test, y_train, y_test = train_test_split(features_scaled, labels_cleaned, test_size=test_size, stratify=labels_cleaned,
                                                            random_state=random_state_global_value)

    print("After Split")
    print("x_train size: " + str(len(x_train)))
    print("x_test size: " + str(len(x_test)))
    print("x_train[0]: ", x_train[0][:10])
    print("x_train[1]: ", x_train[1][:10])
    print("x_train[2]: ", x_train[2][:10])
    print("x_test[0]: ", x_test[0][:10])
    print("x_test[1]: ", x_test[1][:10])
    print("x_test[2]: ", x_test[2][:10])
    oversampling_rate = 0.6
    if dataset_splitting == 1:
        x_train, y_train = balance_dataset(x_train, y_train, balance_method, oversampling_rate)
        print("test")

    # # Standardize features
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train_combined)
    # x_test = scaler.transform(x_test)

    # Scaler Test
    # print("x_train[0]: ", x_train[0][:4])
    # print("x_train[1]: ", x_train[1][:4])
    # print("x_train[2]: ", x_train[2][:4])
    # print("x_test[0]: ", x_test[0][:4])
    # print("x_test[1]: ", x_test[1][:4])
    # print("x_test[2]: ", x_test[2][:4])

    # Test PCA
    # print("Before PCA")
    print("original_x_train: " + str(x_train.shape))
    print("original_x_test: " + str(x_test.shape))
    # print("x_train[0]: ", len(x_train[0]))

    # ALTERNATIVE: Specify the amount of variance to retain
    variance_ratio = 0.95  # Retain 95% of the variance
    pca = PCA(n_components=variance_ratio)
    reduced_x_train = pca.fit_transform(x_train)        # TODO CHECK PCA AGAIN CAUSE THE DATA IS NO LONGER SCALED
    reduced_x_test = pca.transform(x_test)

    print("After PCA")
    print("reduced_x_train: " + str(reduced_x_train.shape))
    print("reduced_x_test: " + str(reduced_x_test.shape))
    # print("reduced_x_train[0]: ", len(reduced_x_train[0]))
    print("reduced_x_train[0]: ", reduced_x_train[0][:10])
    print("reduced_x_train[1]: ", reduced_x_train[1][:10])
    print("reduced_x_train[2]: ", reduced_x_train[2][:10])
    print("reduced_x_test[0]: ", reduced_x_test[0][:10])
    print("reduced_x_test[1]: ", reduced_x_test[1][:10])
    print("reduced_x_test[2]: ", reduced_x_test[2][:10])

    # Balance Test Dataset
    # ONLY NECESSARY WITH OLD METHOD
    if train_test_split_method == "old_method":
        # Balance the test set using RandomUnderSampler
        rus = RandomUnderSampler(sampling_strategy='auto', random_state=random_state_global_value)
        reduced_x_test, y_test = rus.fit_resample(reduced_x_test, y_test)

    # Check Train Dataset Sample Distribution
    class_0_sample_count = sum(y_train == 0)
    class_1_sample_count = sum(y_train == 1)
    print("Total Samples Train:", class_0_sample_count + class_1_sample_count)
    print("Class 0:", class_0_sample_count)
    print("Class 1:", class_1_sample_count)

    # Check Test Dataset Sample Distribution
    class_0_sample_count = sum(y_test == 0)
    class_1_sample_count = sum(y_test == 1)
    print("Total Samples Test:", class_0_sample_count + class_1_sample_count)
    print("Class 0:", class_0_sample_count)
    print("Class 1:", class_1_sample_count)

    # Initialize results

    parameters = {
        'train_samples_number': str(len(x_train)),
        'train_samples_reduced_number': str(len(reduced_x_train)),
        'test_samples_number': str(len(x_test)),
        'test_samples_reduced_number': str(len(reduced_x_test)),
        'test_size %': test_size * 100,
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

    # Voting Classifier Voting Type
    # # Methods: "soft", "hard"
    voting_type = "hard"

    # Initialize all models
    # LR
    if models_used[0] == 1:
        # lr_hyper refers to the hyperparameters for lr
        # first slot is C, the regularization strength
        # Syntax: [a,b,c], Usage: 'C': np.logspace(a, b, c)

        lr_hyper = [[-7, 7, 15]]
        if dataset_splitting > 1:
            # Split the dataset for ensemble learning
            x_train_dataset_split, y_train_dataset_split = training_ensemble_split(reduced_x_train, y_train,
                                                                                   dataset_splitting)

            model_lr = []
            model_lr_hyper = []

            for i in range(dataset_splitting):
                # Training
                model_lr_temp, model_lr_hyper_temp = training.lr_training(x_train_dataset_split[i],
                                                                          y_train_dataset_split[i], lr_hyper, random_state_global_value)
                model_lr.append(model_lr_temp)
                model_lr_hyper.append(model_lr_hyper_temp)

                # Basic statistics Comparison
                x_train_dataset_split_df = pd.DataFrame(x_train_dataset_split[i])
                reduced_x_test_df = pd.DataFrame(reduced_x_test)
                train_stats = x_train_dataset_split_df.describe()
                test_stats = reduced_x_test_df.describe()

                # Display the basic statistics
                print("Training Data Statistics for ensemble model ", i, ":\n", train_stats)
                print("Test Data Statistics:\n", test_stats)

                # Perform t-tests for each feature
                print("T_Tests")
                for column in range(50):
                    t_stat, p_value = ttest_ind(x_train_dataset_split_df[column], reduced_x_test_df[column])
                    print(f'{column}: t-statistic = {t_stat}, p-value = {p_value}')

            # # Training
            # model_lr_1, model_lr_hyper_1 = training.lr_training(x1, y1, lr_hyper)
            # model_lr_2, model_lr_hyper_2 = training.lr_training(x2, y2, lr_hyper)
            # model_lr_3, model_lr_hyper_3 = training.lr_training(x3, y3, lr_hyper)

            estimators_temp = []
            for i in range(dataset_splitting):
                estimators_temp.append(('lr' + str(i + 1), model_lr[i]))

            print(estimators_temp)

            # Create a voting classifier
            ensemble = VotingClassifier(
                estimators=estimators_temp, voting=voting_type)

            ensemble.fit(reduced_x_test, y_test)

            # Evaluating
            if voting_type == "soft":
                y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
                performance_metrics_lr = training.evaluate_pred_proba_model(y_test, y_pred_proba)
            elif voting_type == "hard":
                y_pred = ensemble.predict(reduced_x_test)
                performance_metrics_lr = training.evaluate_pred_model(y_test, y_pred)
                performance_metrics_lr.append(0)
            else:
                # Default voting type = "soft"
                print("Wrong Voting Type Detected, defaulting to soft")
                y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
                performance_metrics_lr = training.evaluate_pred_proba_model(y_test, y_pred_proba)

            # Access individual model predictions
            for name, clf in ensemble.named_estimators_.items():
                # clf_preds = clf.predict(reduced_x_test)
                # print(f"Predictions from {name}: {clf_preds}")

                # Evaluating
                if voting_type == "soft":
                    y_pred_proba = clf.predict_proba(reduced_x_test)[:, 1]
                    performance_metrics_lr_individual = training.evaluate_pred_proba_model(y_test, y_pred_proba)
                elif voting_type == "hard":
                    y_pred = clf.predict(reduced_x_test)
                    performance_metrics_lr_individual = training.evaluate_pred_model(y_test, y_pred)
                else:
                    # Default voting type = "soft"
                    print("Wrong Voting Type Detected, defaulting to soft")
                    y_pred_proba = clf.predict_proba(reduced_x_test)[:, 1]
                    performance_metrics_lr_individual = training.evaluate_pred_proba_model(y_test, y_pred_proba)

                print(f"Predictions from {name}:")
                print("Specificity: " + str(performance_metrics_lr_individual[0]))
                print("Sensitivity: " + str(performance_metrics_lr_individual[1]))
                print("Precision: " + str(performance_metrics_lr_individual[2]))
                print("Accuracy: " + str(performance_metrics_lr_individual[3]))
                print("F1: " + str(performance_metrics_lr_individual[4]))
                if voting_type == "soft":
                    print("AUC: " + str(performance_metrics_lr_individual[5]))

            # Saving the best hyperparameters
            results_model['Hyper_LR__C'] = model_lr_hyper[0]["C"]
            results_model['performance_metrics_lr'] = performance_metrics_lr
        else:
            # Training
            model_lr, model_lr_hyper = training.lr_training(reduced_x_train, y_train, lr_hyper)

            # Evaluating
            y_pred_proba = model_lr.predict_proba(reduced_x_test)[:, 1]
            performance_metrics_lr = training.evaluate_pred_proba_model(y_test, y_pred_proba)

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
            # Split the dataset for ensemble learning
            x_train_dataset_split, y_train_dataset_split = training_ensemble_split(reduced_x_train, y_train,
                                                                                   dataset_splitting)
            model_knn = []
            model_knn_hyper = []

            for i in range(dataset_splitting):
                # Training
                model_knn_temp, model_knn_hyper_temp = training.knn_training(x_train_dataset_split[i],
                                                                             y_train_dataset_split[i], knn_hyper)
                model_knn.append(model_knn_temp)
                model_knn_hyper.append(model_knn_hyper_temp)

            estimators_temp = []
            for i in range(dataset_splitting):
                estimators_temp.append(('knn' + str(i + 1), model_knn[i]))

            # Create a voting classifier
            ensemble = VotingClassifier(
                estimators=estimators_temp, voting=voting_type)

            ensemble.fit(reduced_x_test, y_test)

            # Make predictions and evaluate the ensemble model
            if voting_type == "soft":
                y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
                performance_metrics_knn = training.evaluate_pred_proba_model(y_test, y_pred_proba)
            elif voting_type == "hard":
                y_pred = ensemble.predict(reduced_x_test)
                performance_metrics_knn = training.evaluate_pred_model(y_test, y_pred)
                performance_metrics_knn.append(0)
            else:
                # Default voting type = "soft"
                print("Wrong Voting Type Detected, defaulting to soft")
                y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
                performance_metrics_knn = training.evaluate_pred_proba_model(y_test, y_pred_proba)

            # Access individual model predictions
            for name, clf in ensemble.named_estimators_.items():
                # clf_preds = clf.predict(reduced_x_test)
                # print(f"Predictions from {name}: {clf_preds}")

                # Evaluating
                if voting_type == "soft":
                    y_pred_proba = clf.predict_proba(reduced_x_test)[:, 1]
                    performance_metrics_knn_individual = training.evaluate_pred_proba_model(y_test, y_pred_proba)
                elif voting_type == "hard":
                    y_pred = clf.predict(reduced_x_test)
                    performance_metrics_knn_individual = training.evaluate_pred_model(y_test, y_pred)
                else:
                    # Default voting type = "soft"
                    print("Wrong Voting Type Detected, defaulting to soft")
                    y_pred_proba = clf.predict_proba(reduced_x_test)[:, 1]
                    performance_metrics_knn_individual = training.evaluate_pred_proba_model(y_test, y_pred_proba)

                print(f"Predictions from {name}:")
                print("Specificity: " + str(performance_metrics_knn_individual[0]))
                print("Sensitivity: " + str(performance_metrics_knn_individual[1]))
                print("Precision: " + str(performance_metrics_knn_individual[2]))
                print("Accuracy: " + str(performance_metrics_knn_individual[3]))
                print("F1: " + str(performance_metrics_knn_individual[4]))
                if voting_type == "soft":
                    print("AUC: " + str(performance_metrics_knn_individual[5]))

            # TODO Find a better way to show hyperparameters for ensemble
            # Saving the best hyperparameters
            results_model['Hyper_kNN__n_neighbors'] = model_knn_hyper[0]["n_neighbors"]
            results_model['Hyper_kNN__leaf_size'] = model_knn_hyper[0]["leaf_size"]
            results_model['performance_metrics_knn'] = performance_metrics_knn
        else:
            # Training
            model_knn, model_knn_hyper = training.knn_training(reduced_x_train, y_train, knn_hyper)

            # Evaluating
            y_pred_proba = model_knn.predict_proba(reduced_x_test)[:, 1]
            performance_metrics_knn = training.evaluate_pred_proba_model(y_test, y_pred_proba)

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
        if dataset_splitting > 1:
            # Split the dataset for ensemble learning
            x_train_dataset_split, y_train_dataset_split = training_ensemble_split(reduced_x_train, y_train,
                                                                                   dataset_splitting)
            model_svm = []
            model_svm_hyper = []

            for i in range(dataset_splitting):
                # Training
                model_svm_temp, model_svm_hyper_temp = training.svm_training(x_train_dataset_split[i],
                                                                             y_train_dataset_split[i], svm_hyper)
                model_svm.append(model_svm_temp)
                model_svm_hyper.append(model_svm_hyper_temp)

            estimators_temp = []
            for i in range(dataset_splitting):
                estimators_temp.append(('svm' + str(i + 1), model_svm[i]))

            # Create a voting classifier
            ensemble = VotingClassifier(
                estimators=estimators_temp, voting=voting_type)

            ensemble.fit(reduced_x_test, y_test)

            # Evaluating
            if voting_type == "soft":
                y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
                performance_metrics_svm = training.evaluate_pred_proba_model(y_test, y_pred_proba)
            elif voting_type == "hard":
                y_pred = ensemble.predict(reduced_x_test)
                performance_metrics_svm = training.evaluate_pred_model(y_test, y_pred)
                performance_metrics_svm.append(0)
            else:
                # Default voting type = "soft"
                print("Wrong Voting Type Detected, defaulting to soft")
                y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
                performance_metrics_svm = training.evaluate_pred_proba_model(y_test, y_pred_proba)

            # Access individual model predictions
            for name, clf in ensemble.named_estimators_.items():
                # clf_preds = clf.predict(reduced_x_test)
                # print(f"Predictions from {name}: {clf_preds}")

                # Evaluating
                if voting_type == "soft":
                    y_pred_proba = clf.predict_proba(reduced_x_test)[:, 1]
                    performance_metrics_svm_individual = training.evaluate_pred_proba_model(y_test, y_pred_proba)
                elif voting_type == "hard":
                    y_pred = clf.predict(reduced_x_test)
                    performance_metrics_svm_individual = training.evaluate_pred_model(y_test, y_pred)
                else:
                    # Default voting type = "soft"
                    print("Wrong Voting Type Detected, defaulting to soft")
                    y_pred_proba = clf.predict_proba(reduced_x_test)[:, 1]
                    performance_metrics_svm_individual = training.evaluate_pred_proba_model(y_test, y_pred_proba)

                print(f"Predictions from {name}:")
                print("Specificity: " + str(performance_metrics_svm_individual[0]))
                print("Sensitivity: " + str(performance_metrics_svm_individual[1]))
                print("Precision: " + str(performance_metrics_svm_individual[2]))
                print("Accuracy: " + str(performance_metrics_svm_individual[3]))
                print("F1: " + str(performance_metrics_svm_individual[4]))
                if voting_type == "soft":
                    print("AUC: " + str(performance_metrics_svm_individual[5]))

            # Saving the best hyperparameters
            results_model['Hyper_SVM__C'] = model_svm_hyper[0]["C"]
            results_model['Hyper_SVM__gamma'] = model_svm_hyper[0]["gamma"]
            results_model['performance_metrics_svm'] = performance_metrics_svm
        else:
            # Training
            model_svm, model_svm_hyper = training.svm_training(reduced_x_train, y_train, svm_hyper)

            # Evaluating
            y_pred_proba = model_svm.predict_proba(reduced_x_test)[:, 1]
            performance_metrics_svm = training.evaluate_pred_proba_model(y_test, y_pred_proba)

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
        if dataset_splitting > 1:
            # Split the dataset for ensemble learning
            x_train_dataset_split, y_train_dataset_split = training_ensemble_split(reduced_x_train, y_train,
                                                                                   dataset_splitting)
            model_mlp = []
            model_mlp_hyper = []

            for i in range(dataset_splitting):
                # Training
                model_mlp_temp, model_mlp_hyper_temp = training.mlp_training(x_train_dataset_split[i],
                                                                             y_train_dataset_split[i], mlp_hyper)
                model_mlp.append(model_mlp_temp)
                model_mlp_hyper.append(model_mlp_hyper_temp)

            estimators_temp = []
            for i in range(dataset_splitting):
                estimators_temp.append(('mlp' + str(i + 1), model_mlp[i]))

            # Create a voting classifier
            ensemble = VotingClassifier(
                estimators=estimators_temp, voting=voting_type)

            ensemble.fit(reduced_x_test, y_test)

            # Evaluating
            if voting_type == "soft":
                y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
                performance_metrics_mlp = training.evaluate_pred_proba_model(y_test, y_pred_proba)
            elif voting_type == "hard":
                y_pred = ensemble.predict(reduced_x_test)
                performance_metrics_mlp = training.evaluate_pred_model(y_test, y_pred)
                performance_metrics_mlp.append(0)
            else:
                # Default voting type = "soft"
                print("Wrong Voting Type Detected, defaulting to soft")
                y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
                performance_metrics_mlp = training.evaluate_pred_proba_model(y_test, y_pred_proba)

            # Access individual model predictions
            for name, clf in ensemble.named_estimators_.items():
                # clf_preds = clf.predict(reduced_x_test)
                # print(f"Predictions from {name}: {clf_preds}")

                # Evaluating
                if voting_type == "soft":
                    y_pred_proba = clf.predict_proba(reduced_x_test)[:, 1]
                    performance_metrics_mlp_individual = training.evaluate_pred_proba_model(y_test, y_pred_proba)
                elif voting_type == "hard":
                    y_pred = clf.predict(reduced_x_test)
                    performance_metrics_mlp_individual = training.evaluate_pred_model(y_test, y_pred)
                else:
                    # Default voting type = "soft"
                    print("Wrong Voting Type Detected, defaulting to soft")
                    y_pred_proba = clf.predict_proba(reduced_x_test)[:, 1]
                    performance_metrics_mlp_individual = training.evaluate_pred_proba_model(y_test, y_pred_proba)

                print(f"Predictions from {name}:")
                print("Specificity: " + str(performance_metrics_mlp_individual[0]))
                print("Sensitivity: " + str(performance_metrics_mlp_individual[1]))
                print("Precision: " + str(performance_metrics_mlp_individual[2]))
                print("Accuracy: " + str(performance_metrics_mlp_individual[3]))
                print("F1: " + str(performance_metrics_mlp_individual[4]))
                if voting_type == "soft":
                    print("AUC: " + str(performance_metrics_mlp_individual[5]))

            # Saving the best hyperparameters
            results_model['Hyper_MLP__hidden_layer_sizes'] = model_mlp_hyper[0]["hidden_layer_sizes"]
            results_model['Hyper_MLP__alpha'] = model_mlp_hyper[0]["alpha"]
            results_model['Hyper_MLP__learning_rate_init'] = model_mlp_hyper[0]["learning_rate_init"]
            results_model['performance_metrics_mlp'] = performance_metrics_mlp

            # # Monitor Convergence through the Loss Curve
            # # Evaluate the best model on the test set
            # y_pred = model_mlp.predict(x_test)
            # accuracy = accuracy_score(y_test, y_pred)
            # print(f'Accuracy: {accuracy}')
            #
            # # Plot the loss curve to monitor convergence
            # plt.plot(model_mlp.loss_curve_)
            # plt.title('Loss Curve')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.grid(True)
            # plt.show()

        else:
            # Training
            model_mlp, model_mlp_hyper = training.mlp_training(reduced_x_train, y_train, mlp_hyper)

            # Evaluating
            y_pred_proba = model_mlp.predict_proba(reduced_x_test)[:, 1]
            performance_metrics_mlp = training.evaluate_pred_proba_model(y_test, y_pred_proba)

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
        if dataset_splitting > 1:
            # Split the dataset for ensemble learning
            x_train_dataset_split, y_train_dataset_split = training_ensemble_split(reduced_x_train, y_train,
                                                                                   dataset_splitting)
            model_cnn = []
            model_cnn_hyper = []

            for i in range(dataset_splitting):
                # Training
                model_cnn_temp, model_cnn_hyper_temp = training.cnn_training(x_train_dataset_split[i],
                                                                             y_train_dataset_split[i], cnn_hyper)
                model_cnn.append(model_cnn_temp)
                model_cnn_hyper.append(model_cnn_hyper_temp)

            estimators_temp = []
            for i in range(dataset_splitting):
                estimators_temp.append(('cnn' + str(i + 1), model_cnn[i]))

            # Create a voting classifier
            ensemble = VotingClassifier(
                estimators=estimators_temp, voting=voting_type)

            ensemble.fit(reduced_x_test, y_test)

            # Evaluating
            if voting_type == "soft":
                y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
                performance_metrics_cnn = training.evaluate_pred_proba_model(y_test, y_pred_proba)
            elif voting_type == "hard":
                y_pred = ensemble.predict(reduced_x_test)
                performance_metrics_cnn = training.evaluate_pred_model(y_test, y_pred)
                performance_metrics_cnn.append(0)
            else:
                # Default voting type = "soft"
                print("Wrong Voting Type Detected, defaulting to soft")
                y_pred_proba = ensemble.predict_proba(reduced_x_test)[:, 1]  # Probability of the positive class
                performance_metrics_cnn = training.evaluate_pred_proba_model(y_test, y_pred_proba)

            # Access individual model predictions
            for name, clf in ensemble.named_estimators_.items():
                # clf_preds = clf.predict(reduced_x_test)
                # print(f"Predictions from {name}: {clf_preds}")

                # Evaluating
                if voting_type == "soft":
                    y_pred_proba = clf.predict_proba(reduced_x_test)[:, 1]
                    performance_metrics_cnn_individual = training.evaluate_pred_proba_model(y_test, y_pred_proba)
                elif voting_type == "hard":
                    y_pred = clf.predict(reduced_x_test)
                    performance_metrics_cnn_individual = training.evaluate_pred_model(y_test, y_pred)
                else:
                    # Default voting type = "soft"
                    print("Wrong Voting Type Detected, defaulting to soft")
                    y_pred_proba = clf.predict_proba(reduced_x_test)[:, 1]
                    performance_metrics_cnn_individual = training.evaluate_pred_proba_model(y_test, y_pred_proba)

                print(f"Predictions from {name}:")
                print("Specificity: " + str(performance_metrics_cnn_individual[0]))
                print("Sensitivity: " + str(performance_metrics_cnn_individual[1]))
                print("Precision: " + str(performance_metrics_cnn_individual[2]))
                print("Accuracy: " + str(performance_metrics_cnn_individual[3]))
                print("F1: " + str(performance_metrics_cnn_individual[4]))
                if voting_type == "soft":
                    print("AUC: " + str(performance_metrics_cnn_individual[5]))

            # Saving the best hyperparameters
            results_model['Hyper_CNN__num_filters'] = model_cnn_hyper[0]["num_filters"]
            results_model['Hyper_CNN__kernel_size'] = model_cnn_hyper[0]["kernel_size"]
            results_model['Hyper_CNN__dropout_rate'] = model_cnn_hyper[0]["dropout_rate"]
            results_model['Hyper_CNN__dense_size'] = model_cnn_hyper[0]["dense_size"]
            results_model['Hyper_CNN__batch_size'] = model_cnn_hyper[0]["batch_size"]
            results_model['Hyper_CNN__epochs'] = model_cnn_hyper[0]["epochs"]

            results_model['performance_metrics_cnn'] = performance_metrics_cnn
        else:
            # Training
            model_cnn, model_cnn_hyper = training.cnn_training(reduced_x_train, y_train, cnn_hyper)

            # Evaluating
            y_pred_proba = model_cnn.predict_proba(reduced_x_test)[:, 1]
            performance_metrics_cnn = training.evaluate_pred_proba_model(y_test, y_pred_proba)

            # Saving the best hyperparameters
            results_model['Hyper_CNN__num_filters'] = model_cnn_hyper["num_filters"]
            results_model['Hyper_CNN__kernel_size'] = model_cnn_hyper["kernel_size"]
            results_model['Hyper_CNN__dropout_rate'] = model_cnn_hyper["dropout_rate"]
            results_model['Hyper_CNN__dense_size'] = model_cnn_hyper["dense_size"]
            results_model['Hyper_CNN__batch_size'] = model_cnn_hyper["batch_size"]
            results_model['Hyper_CNN__epochs'] = model_cnn_hyper["epochs"]

            results_model['performance_metrics_cnn'] = performance_metrics_cnn

    return results_model, parameters


# Convert models_used values to their names
def models_name_switch_table(model):
    switcher = {
        0: "performance_metrics_lr",
        1: "performance_metrics_knn",
        2: "performance_metrics_svm",
        3: "performance_metrics_mlp",
        4: "performance_metrics_cnn",
        5: "performance_metrics_lstm"
    }

    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return switcher.get(model, "Model Name Mismatch")


# Function to store the results of each individual iteration
def save_iteration_csv(results_df, models_used_str, parameters_df, iteration_identifier):
    for perf_res in models_used_str:
        if perf_res in results_df:
            # print(perf_res)
            metrics = results_df[perf_res]
            # print(metrics.apply(lambda x: [f"{num:.4f}" for num in x]))

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
            parameters_df.to_csv(
                './' + data_dir_choice + '/model_metrics/my_dataframe_expanded_' + str(iteration_identifier) + '.csv',
                index=False)

            print("Iteration saved")


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
            parameters_df.to_csv(
                './' + data_dir_choice + '/model_metrics/my_dataframe_expanded_' + current_date + '.csv',
                index=False)


results_df = modular_model_training()
