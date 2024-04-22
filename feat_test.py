# Testing the feature extraction functions
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Dataset directory
data_dir = "E:/Storage/University/Thesis/smarty4covid/"


# Feature extraction function with only one method and only one hyperparameter
def test_extract_features_simple(data_dir, audio_paths_part_1, audio_paths_part_2, n_mfcc):
    file_path = os.path.join(data_dir, audio_paths_part_1, audio_paths_part_2, "audio.cough.mp3")
    # Check if the directory exists, if not, create it
    if os.path.exists(file_path):
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)

        # Save MFCCs in a NumPy array
        mfccs_array = np.array(mfccs)

        return mfccs_processed, mfccs_array
    return False, False


# test
def test_feat_extr_init(k_mfcc):
    data_index = os.path.join(data_dir, 'smarty4covid_tabular_data.csv')
    data = pd.read_csv(data_index)

    # Exclude rows where 'covid_status' is 'no'
    data = data[data.covid_status != 'no']

    # Initialize the LabelEncoder
    le = LabelEncoder()
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

    test_mfccs_target = "extracted_mfccs_" + str(k_mfcc) + ".npy"
    test_mfccs_filename = os.path.join(features_folder, test_mfccs_target)

    successful_indices = []
    features_list = []

    # test_mfccs_list = []
    # Initialize an empty list to store the padded/truncated MFCCs
    padded_mfccs_list = []

    # Check if the file doesn't exist
    if not os.path.exists(feature_filename):
        # Modified part to extract features and simultaneously filter labels
        for idx, row in data.iterrows():
            feat, mfccs = test_extract_features_simple(data_dir, row.participantid, row.submissionid, n_mfcc)
            if feat is not False:
                features_list.append(feat)
                successful_indices.append(idx)

                #test_mfccs_list.append(mfccs)

                # Pad or truncate the MFCCs to a fixed length (e.g., 100 frames)
                max_frames = 100
                if mfccs.shape[1] < max_frames:
                    # If the number of frames is less than max_frames, pad the MFCCs
                    padded_mfccs = np.pad(mfccs, ((0, 0), (0, max_frames - mfccs.shape[1])), mode='constant')
                else:
                    # If the number of frames is greater than or equal to max_frames, truncate the MFCCs
                    padded_mfccs = mfccs[:, :max_frames]

                padded_mfccs_list.append(padded_mfccs)

                # Convert the list of padded/truncated MFCCs to a NumPy array
            padded_mfccs_array = np.array(padded_mfccs_list)



        #test_mfccs = np.array(test_mfccs_list)

        features = np.array(features_list)

        # Filter labels based on successful feature extraction
        labels = np.array(data.loc[successful_indices, 'covid_status'])

        # Convert labels to a consistent numerical format
        labels = le.fit_transform(labels)

        np.save(feature_filename, features)
        np.save(label_filename, labels)

        np.save(test_mfccs_filename, padded_mfccs_array)


kmfcc = 5
# test_feat_extr_init(kmfcc)

# testi = "E:/Storage/Work/thesisProjectTest/extracted_features/feat_extr_simple/extracted_mfccs_5.npy"
# test = np.load(testi)
# print("size" + str(np.size(testi[0])))
# print(test)


# ultimate test for one sample only
def test_one_sample_mfcc():
    k_mfcc = 5

    audio_paths_part_1 = "0008333a-8df4-4c31-8af0-9f34c7e8720d"
    audio_paths_part_2 = "fd70fad9-925f-4948-8ca4-bcb4b9874cae"
    n_mfcc = 14 * k_mfcc
    mfccs_processed, mfccs_array = test_extract_features_simple(data_dir, audio_paths_part_1, audio_paths_part_2, n_mfcc)

    print("Mfccs Raw")
    print("size of the entire mfcc list: " + str(np.size(mfccs_array)))
    print("size of one mfcc frame: " + str(np.size(mfccs_array[0])))

    max_mfcc_0 = max(mfccs_array[0])
    formatted_number = "{:e}".format(max_mfcc_0)

    print("max mfcc: " + str(formatted_number))
    print(mfccs_array)

    print("Mfccs Processed")
    # Save MFCCs in a NumPy array
    mfccs_processed_array = np.array(mfccs_processed)
    print("size of the entire mfcc list: " + str(np.size(mfccs_processed_array)))

    max_mfcc_0 = np.max(mfccs_processed_array)
    formatted_number = "{:e}".format(max_mfcc_0)

    print("max mfcc: " + str(formatted_number))
    print(mfccs_processed_array)

    print("Final Mfccs Processed")
    final_mfccs_processed_filename = "E:/Storage/Work/thesisProjectTest/extracted_features/feat_extr_simple/extracted_features_5.npy"
    final_mfccs_processed = np.load(final_mfccs_processed_filename)
    print("size of the entire mfcc list: " + str(np.size(final_mfccs_processed)))
    print("size of one mfcc frame: " + str(np.size(final_mfccs_processed[0])))

    max_mfcc_0 = np.max(final_mfccs_processed[0])
    formatted_number = "{:e}".format(max_mfcc_0)

    print("max mfcc: " + str(formatted_number))

    print(final_mfccs_processed)


test_one_sample_mfcc()
