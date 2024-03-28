import os
import pandas as pd
import numpy as np

import feature_extraction as feat_extr
import training

# IMPORTANT, if run on different PCs, this needs to be changed to point to the dataset directory
# Every dataset path query is formed in relation to this variable (data_dir)
# Dataset directory
data_dir = "E:/Storage/University/Thesis/smarty4covid/"


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


# Feature Extraction Initialization Function with all five methods and only two hyperparameters
def feat_extr_init(data_dir):
    # Load the index csv
    data_index = os.path.join(data_dir, 'smarty4covid_tabular_data.csv')
    data = pd.read_csv(data_index)

    # Hyperparameters based on:
    # Preliminary diagnosis of COVID-19 based on cough sounds using machine learning algorithms
    # https://ieeexplore.ieee.org/abstract/document/9432324
    k_values_mfcc = [1, 2, 3, 4, 5]
    k_values_frame = [8, 9, 10, 11, 12]

    # Initialize list for storing results
    results = []

    # Loop over all combinations of hyperparameters
    for k_mfcc in k_values_mfcc:
        n_mfcc = 14 * k_mfcc
        for k_frame in k_values_frame:
            frame_size = 2 ** k_frame
            hop_length = frame_size // 2  # 50% overlap

            # Name of the directory and file where the features will be saved
            features_folder = "extracted_features/feat_extr"

            # Check if the directory exists, if not, create it
            if not os.path.exists(features_folder):
                os.makedirs(features_folder)

            feature_filename_target = "extracted_features_" + str(k_mfcc) + str(k_frame) + ".npy"
            feature_filename = os.path.join(features_folder, feature_filename_target)

            # Check if the file already exists
            if os.path.exists(feature_filename):
                # Load the features from the file
                features = np.load(feature_filename)
            else:
                # Extract features because they don't exist
                features = np.array([feat_extr.extract_features(data_dir, row.participantid, row.submissionid, n_mfcc, frame_size, hop_length) for idx, row in data.iterrows()])
                # Save the extracted features to the file for future use
                np.save(feature_filename, features)

            # Labels
            labels = np.array(data.covid_status)

            # Train and evaluate the different classifiers outlined in training.py
            results.append(training.classifier(features, labels, n_mfcc, frame_size))

        print("Current Hyperparameters Complete: k_mfcc = " + str(k_mfcc))
    print("Process Complete")

    # After the loop you can convert results to a DataFrame and analyze it
    results_df = pd.DataFrame(results)


# Feature Extraction Initialization Function with all five methods and all three hyperparameters
def feat_extr_with_segm_init(data_dir):
    # Load the index csv
    data_index = os.path.join(data_dir, 'smarty4covid_tabular_data.csv')
    data = pd.read_csv(data_index)

    # Hyperparameters based on:
    # Preliminary diagnosis of COVID-19 based on cough sounds using machine learning algorithms
    # https://ieeexplore.ieee.org/abstract/document/9432324
    k_values_mfcc = [1, 2, 3, 4, 5]
    k_values_frame = [8, 9, 10, 11, 12]
    k_values_segment = [5, 7, 10, 12, 15]

    # Initialize list for storing results
    results = []

    # Loop over all combinations of hyperparameters
    for k_mfcc in k_values_mfcc:
        n_mfcc = 14 * k_mfcc
        for k_frame in k_values_frame:
            frame_size = 2 ** k_frame
            hop_length = frame_size // 2  # 50% overlap
            for k_segment in k_values_segment:
                n_segments = 10 * k_segment

                # Name of the directory and file where the features will be saved
                features_folder = "extracted_features/feat_extr_with_segm"

                # Check if the directory exists, if not, create it
                if not os.path.exists(features_folder):
                    os.makedirs(features_folder)

                feature_filename_target = "extracted_features_" + str(k_mfcc) + str(k_frame) + str(k_segment) + ".npy"
                feature_filename = os.path.join(features_folder, feature_filename_target)

                # Check if the file already exists
                if os.path.exists(feature_filename):
                    # Load the features from the file
                    features = np.load(feature_filename)
                else:
                    # Extract features because they don't exist
                    features = np.array([feat_extr.extract_features_with_segments(data_dir, row.participantid, row.submissionid, n_mfcc, frame_size, hop_length, n_segments) for idx, row in data.iterrows()])
                    # Save the extracted features to the file for future use
                    np.save(feature_filename, features)

                # Labels
                labels = np.array(data.covid_status)

                # Train and evaluate the different classifiers outlined in training.py
                results.append(training.classifier(features, labels, n_mfcc, frame_size, n_segments))

            print("Current Hyperparameters Done: k_mfcc = " + str(k_mfcc) + ", k_frame = " + str(k_frame))
        print("Current Hyperparameters Complete: k_mfcc = " + str(k_mfcc))
    print("Process Complete")

    # After the loop you can convert results to a DataFrame and analyze it
    results_df = pd.DataFrame(results)
