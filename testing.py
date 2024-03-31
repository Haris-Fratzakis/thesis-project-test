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


# Testing function for feature_extraction_simple and LR
# Feature Extraction Initialization Function with only one method and only one hyperparameter
def feat_extr_simple_test(data_dir):
    # Load the index csv
    data_index = os.path.join(data_dir, 'smarty4covid_tabular_data.csv')
    data = pd.read_csv(data_index)

    # Exclude rows where 'covid_status' is 'no'
    data = data[data.covid_status != 'no']

    # Initialize the LabelEncoder
    le = LabelEncoder()

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
        label_filename_target = "extracted_labels_" + str(k_mfcc) + ".npy"
        feature_filename = os.path.join(features_folder, feature_filename_target)
        label_filename = os.path.join(features_folder, label_filename_target)

        successful_indices = []
        features_list = []
        # Check if the file doesn't exist
        if os.path.exists(feature_filename):
            features = np.load(feature_filename)
            labels = np.load(label_filename)
        else:
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

        # Train and evaluate the different classifiers outlined in training.py
        results.append(test_classifier(features, labels, n_mfcc))

    print("Process Complete")

    # After the loop you can convert results to a DataFrame and analyze it
    results_df = pd.DataFrame(results)
    return results_df


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


results_df = feat_extr_simple_test(data_dir)

# Set pandas display options
pd.set_option('display.max_rows', None)  # or replace None with the exact number of rows you expect
pd.set_option('display.max_columns', None)  # or replace None with the exact number of columns you expect
pd.set_option('display.width', 1000)  # Adjust the width to fit your screen if necessary

# Print the entire DataFrame
metrics = results_df['performance_metrics_lr']
print(results_df['performance_metrics_lr'].apply(lambda x: [f"{num:.4f}" for num in x]))


# Convert the 'array_column' to a DataFrame and expand it into separate columns
array_df = pd.DataFrame(results_df['performance_metrics_lr'].tolist(), index=results_df.index)

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
