import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import feature_extraction as feat_extr
import training


# Testing function for feature_extraction_simple and LR
# Feature Extraction Initialization Function with only one method and only one hyperparameter
def feat_extr_simple_init(data_dir):
    # Load the index csv
    data_index = os.path.join(data_dir, 'smarty4covid_tabular_data.csv')
    data = pd.read_csv(data_index)

    # Hyperparameters based on:
    # Preliminary diagnosis of COVID-19 based on cough sounds using machine learning algorithms
    # https://ieeexplore.ieee.org/abstract/document/9432324
    k_values_mfcc = [1, 2, 3, 4, 5]

    # Initialize list for storing results
    results = []

    # Loop over all combinations of hyperparameters
    for k_mfcc in k_values_mfcc:
        n_mfcc = 14 * k_mfcc

        # Name of the directory and file where the features will be saved
        features_folder = "extracted_features/feat_extr_simple"

        # Check if the directory exists, if not, create it
        if not os.path.exists(features_folder):
            os.makedirs(features_folder)

        feature_filename_target = "extracted_features_" + str(k_mfcc) + ".npy"
        feature_filename = os.path.join(features_folder, feature_filename_target)

        # Check if the file already exists
        if os.path.exists(feature_filename):
            # Load the features from the file
            features = np.load(feature_filename)
        else:
            # Extract features because they don't exist
            features = np.array([feat_extr.extract_features_simple(data_dir, row.participantid, row.submissionid, n_mfcc) for idx, row in data.iterrows()])
            # Save the extracted features to the file for future use
            np.save(feature_filename, features)

        # Labels
        labels = np.array(data.covid_status)

        # Train and evaluate the different classifiers outlined in training.py
        results.append(training.classifier(features, labels, n_mfcc))

    print("Process Complete")

    # After the loop you can convert results to a DataFrame and analyze it
    results_df = pd.DataFrame(results)


# Test training function
def test_classifier(features, labels, n_mfcc, frame_size=0, n_segments=0):
    # Splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Initialize all models
    model_lr = training.lr_training(x_train, y_train)

    # Predict and evaluate all models
    # LR
    y_pred_proba = model_lr.predict_proba(x_test)[:, 1]
    performance_metrics_lr = training.evaluate_model(y_test, y_pred_proba)

    # Save results
    results = {
        'mfcc': n_mfcc,
        'frame_size': frame_size,
        'segments': n_segments,
        'performance_metrics_lr': performance_metrics_lr
    }

    return results