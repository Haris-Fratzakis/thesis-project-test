import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

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
    k_values_mfcc = [1, 2, 3, 4, 5]
    test_feat_extr(data, k_values_mfcc)

    # models_used signifies which model is used, each slot signifies a different model
    # 1 means model is going to be used, 0 means it will not be used
    # slots are [LR, KNN, SVM, MLP, CNN, LSTM]
    models_used = [1, 0, 0, 0, 0, 0]
    # This is the modular classifier training stage
    results_df = test_classifier_mod(k_values_mfcc, models_used)

    return results_df


# Temporary feature extraction function
def test_feat_extr(data, k_values_mfcc):
    # Initialize the LabelEncoder
    le = LabelEncoder()

    # Loop over all combinations of hyperparameters
    for k_mfcc in k_values_mfcc:
        n_mfcc = 14 * k_mfcc

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


# Temporary classifier function
def test_classifier_mod(k_values_mfcc, models_used):
    # Initialize list for storing results
    results = []

    # Loop over all combinations of hyperparameters
    for k_mfcc in k_values_mfcc:
        n_mfcc = 14 * k_mfcc

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

    # After the loop you can convert results to a DataFrame and analyze it
    results_df = pd.DataFrame(results)
    return results_df


# Test training function
def test_classifier(features, labels, n_mfcc=-1, frame_size=-1, n_segments=-1, models_used=None):
    if models_used is None:
        models_used = [0, 0, 0, 0, 0, 0]

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Initialize results
    results_model = {
        'mfcc': n_mfcc,
        'frame_size': frame_size,
        'segments': n_segments,
    }

    # Initialize all models
    # LR
    if models_used[0] == 1:
        # lr_hyper refers to the hyperparameters for lr
        # first slot is C, the inverse of regularization strength
        # Syntax: [a,b,c], Usage: 'C': np.logspace(a, b, c)

        lr_hyper = [[-7, 7, 15]]
        # Training
        model_lr = training.lr_training(x_train, y_train, lr_hyper)
        # model_lr = training.lr_training(x_train, y_train)

        # Evaluating
        y_pred_proba = model_lr.predict_proba(x_test)[:, 1]
        performance_metrics_lr = training.evaluate_model(y_test, y_pred_proba)

        results_model['performance_metrics_lr'] = performance_metrics_lr

    # Predict and evaluate all models

    return results_model


# Function to display the results dataframe in a better way
def test_display(results_df):
    # Print the entire DataFrame
    if 'performance_metrics_lr' in results_df:
        metrics = results_df['performance_metrics_lr']
        print(metrics.apply(lambda x: [f"{num:.4f}" for num in x]))

        # Convert the 'array_column' to a DataFrame and expand it into separate columns
        array_df = pd.DataFrame(metrics.tolist(), index=results_df.index)

        # Your specific list of names for the expanded columns
        column_names = ["specificity", "sensitivity", "precision", "accuracy", "F1", "AUC"]

        # Ensure the list length matches the number of columns to rename
        if len(column_names) == array_df.shape[1]:
            array_df.columns = column_names
        else:
            raise ValueError("The number of column names does not match the number of columns.")

        # Join the new columns back with the original DataFrame
        df_expanded = pd.concat([results_df.drop('performance_metrics_lr', axis=1), array_df], axis=1)

        # Save the expanded DataFrame to a CSV file
        df_expanded.to_csv('my_dataframe_expanded.csv', index=False)


results_df = test_modular(data_dir)
test_display(results_df)

# TODO add support for more feature extraction hyperparameters
