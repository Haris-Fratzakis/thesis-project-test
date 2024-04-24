import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from datetime import datetime

import feature_extraction as feat_extr
import training

# IMPORTANT, if run on different PCs, this needs to be changed to point to the dataset directory
# Every dataset path query is formed in relation to this variable (data_dir)
# Dataset directory
data_dir = "E:/Storage/University/Thesis/smarty4covid/"


# New Test function to have modular feature extraction and training functions
def test_modular(data_dir):
    # Load the index csv
    data_index = os.path.join(data_dir, 'smarty4covid_tabular_data.csv')
    data = pd.read_csv(data_index)

    # Exclude rows where 'covid_status' is 'no'
    data = data[data.covid_status != 'no']

    # Further preprocessing can be done here or split to another function

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
    k_values_segment = None
    test_feat_extr(data=data, k_values_mfcc=k_values_mfcc, k_values_frame=k_values_frame, k_values_segment=k_values_segment)

    # models_used signifies which model is used, each slot signifies a different model
    # 1 means model is going to be used, 0 means it will not be used
    # slots are [LR, KNN, SVM, MLP, CNN, LSTM]
    models_used = [0, 1, 0, 0, 0, 0]
    # TODO Fix SVM bug of never converging to a solution
    # This is the modular classifier training stage
    results_df = test_classifier_mod(k_values_mfcc=k_values_mfcc, k_values_frame=k_values_frame, k_values_segment=k_values_segment, models_used=models_used)
    # print(results_df)

    # convert models_used values to their names
    for idx in range(len(models_used)):
        if models_used[idx] == 1:
            models_used[idx] = models_name_conv(idx)

    print("Models Used List: " + str(models_used))
    test_display(results_df, models_used)

    return results_df


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
            features_folder = "extracted_features/feat_extr_simple"

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
                    feat = feat_extr.extract_features_simple(data_dir, row.participantid, row.submissionid, n_mfcc)
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
                    features_folder = "extracted_features/feat_extr"

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
                            # TODO Fix error
                            feat = feat_extr.extract_features(data_dir, row.participantid, row.submissionid, n_mfcc, frame_size, hop_length)
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
                        features_folder = "extracted_features/feat_extr_with_segm"

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
                            # Modified part to extract features and simultaneously filter labels
                            for idx, row in data.iterrows():
                                feat = feat_extr.extract_features_with_segments(data_dir, row.participantid, row.submissionid, n_mfcc, frame_size, hop_length, n_segments)
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


# Temporary classifier function
def test_classifier_mod(k_values_mfcc, k_values_frame=None, k_values_segment=None, models_used=None):
    if k_values_mfcc is None:
        k_values_mfcc = [1]
    if k_values_frame is None:
        k_values_frame = [-1]
    if k_values_segment is None:
        k_values_segment = [-1]
    if models_used is None:
        models_used = [0, 0, 0, 0, 0, 0]

    # Initialize list for storing results
    results = []

    # Loop over all combinations of hyperparameters
    for k_mfcc in k_values_mfcc:
        n_mfcc = 14 * k_mfcc

        # Feature Extraction Initialization Function with only one method and only one hyperparameter
        if k_values_frame == [-1]:
            # Name of the directory and file where the features will be saved
            features_folder = "extracted_features/feat_extr_simple"

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
                results.append(test_classifier(features, labels, n_mfcc=n_mfcc, models_used=models_used))
            else:
                print("Error, data mismatch, features and labels data don't exist in features folder in test classifier")
        else:
            for k_frame in k_values_frame:
                frame_size = 2 ** k_frame
                hop_length = frame_size // 2  # 50% overlap

                # Feature Extraction Initialization Function with all five methods and only two hyperparameters
                if k_values_segment == [-1]:
                    # Name of the directory and file where the features will be saved
                    features_folder = "extracted_features/feat_extr"

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
                        results.append(test_classifier(features, labels, n_mfcc=n_mfcc, frame_size=frame_size, models_used=models_used))
                    else:
                        print(
                            "Error, data mismatch, features and labels data don't exist in features folder in test classifier")
                else:
                    for k_segment in k_values_segment:
                        n_segments = 10 * k_segment

                        # Feature Extraction Initialization Function with all five methods and all three hyperparameters

                        # Name of the directory and file where the features will be saved
                        features_folder = "extracted_features/feat_extr_with_segm"

                        # Check if the directory exists, if not, report error, data mismatch
                        if not os.path.exists(features_folder):
                            print("Error, data mismatch, features folder doesn't exist in test classifier")

                        feature_filename_target = "extracted_features_" + str(k_mfcc) + "_" + str(k_frame) + "_" + str(k_segment) + ".npy"
                        label_filename_target = "extracted_labels_" + str(k_mfcc) + "_" + str(k_frame) + "_" + str(k_segment) + ".npy"
                        feature_filename = os.path.join(features_folder, feature_filename_target)
                        label_filename = os.path.join(features_folder, label_filename_target)

                        # Check if the file doesn't exist
                        if os.path.exists(feature_filename):
                            features = np.load(feature_filename)
                            labels = np.load(label_filename)

                            # Train and evaluate the different classifiers outlined in training.py
                            results.append(test_classifier(features, labels, n_mfcc=n_mfcc, frame_size=frame_size, n_segments=n_segments, models_used=models_used))
                        else:
                            print(
                                "Error, data mismatch, features and labels data don't exist in features folder in test classifier")

    # After the loop you can convert results to a DataFrame and analyze it
    results_df = pd.DataFrame(results)
    return results_df


# Balancing Dataset Function
def balance_dataset(features, labels, balance_method):
    # Check class distribution
    # Class 0: Negative Covid
    # Class 1: Positive Covid
    print("Before balancing:")
    print("Class 0:", sum(labels == 0))
    print("Class 1:", sum(labels == 1))

    match balance_method:
        case "Resampling":
            # Apply Random Over Sampling and Under Sampling
            # Define the resampling strategy
            over = RandomOverSampler(sampling_strategy=0.8)  # Oversample the minority to have 50% of the majority class
            under = RandomUnderSampler(sampling_strategy=1.0)  # Undersample the majority to have equal to the minority

            # First apply oversampling
            features_oversampled, labels_oversampled = over.fit_resample(features, labels)

            print("After oversampling:")
            print("Class 0:", sum(labels_oversampled == 0))
            print("Class 1:", sum(labels_oversampled == 1))

            # Then apply undersampling
            features_combined, labels_combined = under.fit_resample(features_oversampled, labels_oversampled)

            # Check new class distribution
            print("After undersampling:")
            print("Class 0:", sum(labels_combined == 0))
            print("Class 1:", sum(labels_combined == 1))

            return features_combined, labels_combined
        case "SMOTE":
            # Check class distribution before SMOTE
            print("Before SMOTE:")
            print("Class 0:", sum(labels == 0))  # Minority class
            print("Class 1:", sum(labels == 1))  # Majority class

            # Apply SMOTE
            smote = SMOTE(random_state=42)
            features_res, labels_res = smote.fit_resample(features, labels)

            # Check class distribution after SMOTE
            print("After SMOTE:")
            print("Class 0:", sum(labels_res == 0))
            print("Class 1:", sum(labels_res == 1))

            return features_res, labels_res
        case _:
            return False, False


# Test training function
def test_classifier(features, labels, n_mfcc=-1, frame_size=-1, n_segments=-1, models_used=None):
    if models_used is None:
        models_used = [0, 0, 0, 0, 0, 0]
        print("No model specified")

    # Balance dataset
    # Methods: Resampling, SMOTE
    balance_method = "Resampling"
    features_combined, labels_combined = balance_dataset(features, labels, balance_method)

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(features_combined, labels_combined, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Initialize results
    results_model = {
        'dataset_balance_method': balance_method,
        'mfcc': n_mfcc,
        'frame_size': frame_size,
        'segments': n_segments,
    }

    # Initialize all models
    # LR
    if models_used[0] == 1:
        # lr_hyper refers to the hyperparameters for lr
        # first slot is C, the regularization strength
        # Syntax: [a,b,c], Usage: 'C': np.logspace(a, b, c)

        lr_hyper = [[-7, 7, 15]]
        # Training
        model_lr, model_lr_hyper = training.lr_training(x_train, y_train, lr_hyper)
        # model_lr = training.lr_training(x_train, y_train)

        # Evaluating
        y_pred_proba = model_lr.predict_proba(x_test)[:, 1]
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
        # Training
        model_knn, model_knn_hyper = training.knn_training(x_train, y_train, knn_hyper)

        # Evaluating
        y_pred_proba = model_knn.predict_proba(x_test)[:, 1]
        performance_metrics_knn = training.evaluate_model(y_test, y_pred_proba)

        # Saving the best hyperparameters
        results_model['Hyper_kNN__n_neighbors'] = model_knn_hyper["n_neighbors"]
        results_model['Hyper_kNN__leaf_size'] = model_knn_hyper["leaf_size"]

        results_model['performance_metrics_knn'] = performance_metrics_knn

    if models_used[2] == 1:
        # svm_hyper refers to the hyperparameters for svm
        # first slot is C, the regularization strength
        # Syntax: [a,b,c], Usage: 'n_neighbors': list(range(a, b, c)),
        # second slot is gamma, which controls the kernel coefficient
        # Syntax: [a,b,c], Usage: 'C': np.logspace(a, b, c)

        svm_hyper = [[-7, 7, 15], [-7, 7, 15]]
        # Training
        model_svm, model_svm_hyper = training.svm_training(x_train, y_train, svm_hyper)

        # Evaluating
        y_pred_proba = model_svm.predict_proba(x_test)[:, 1]
        performance_metrics_svm = training.evaluate_model(y_test, y_pred_proba)

        # Saving the best hyperparameters
        results_model['Hyper_SVM__C'] = model_svm_hyper["C"]

        results_model['performance_metrics_svm'] = performance_metrics_svm

    # Predict and evaluate all models

    return results_model


# Convert models_used values to their names
def models_name_conv(model):
    switcher = {
        0: "performance_metrics_lr",
        1: "performance_metrics_knn",
        2: "performance_metrics_svm",
    }

    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return switcher.get(model, "Model Name Mismatch")


# Function to display the results dataframe in a better way
def test_display(results_df, models_used_str):
    # TODO Add the hyperparameters of the models to the excel
    # Print the entire DataFrame
    print("Metrics Used: [specificity, sensitivity, precision, accuracy, F1, AUC]")
    for perf_res in models_used_str:
        if perf_res in results_df:
            print(perf_res)
            metrics = results_df[perf_res]
            print(metrics.apply(lambda x: [f"{num:.4f}" for num in x]))

            # Convert the 'array_column' to a DataFrame and expand it into separate columns
            array_df = pd.DataFrame(metrics.tolist(), index=results_df.index)

            # Your specific list of names for the expanded columns
            column_names = [perf_res + "_specificity", perf_res + "_sensitivity", perf_res + "_precision", perf_res + "_accuracy", perf_res + "_F1", perf_res + "_AUC"]

            # Ensure the list length matches the number of columns to rename
            if len(column_names) == array_df.shape[1]:
                array_df.columns = column_names
            else:
                raise ValueError("The number of column names does not match the number of columns.")

            # Join the new columns back with the original DataFrame
            results_df = pd.concat([results_df.drop(perf_res, axis=1), array_df], axis=1)

            metrics_folder = "./model_metrics"

            # Check if the directory exists, if not, create it
            if not os.path.exists(metrics_folder):
                os.makedirs(metrics_folder)

            # Get the current date
            current_date = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

            # Save the expanded DataFrame to a CSV file
            results_df.to_csv('./model_metrics/my_dataframe_expanded_' + current_date + "_" + results_df["dataset_balance_method"] + '.csv', index=False)


results_df = test_modular(data_dir)
