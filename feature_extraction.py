import os
import librosa
import numpy as np
import scipy.stats
from scipy.stats import mode


# Feature Extraction Methods Used:
#   Mel Frequency Cepstral Coefficients(MFCCs)
#   Spectral Centroid (SC)
#   Spectral Roll-off(SR)
#   Zero-Crossing rate (ZCR)
#   Kurtosis

# Feature Extraction Methods based on Preliminary diagnosis of COVID-19 based on cough sounds using machine learning algorithms
# https://ieeexplore.ieee.org/abstract/document/9432324

# Feature extraction function with only one method and only one hyperparameter
def extract_features_simple(data_dir, audio_path, audio_name, n_mfcc):
    file_path = os.path.join(data_dir, audio_path, audio_name)
    # Check if the directory exists, if not, create it
    if os.path.exists(file_path):
        if os.path.getsize(file_path) <= 2048:
            # print(f"The audio file {file_path} is empty. Skipping this file.")
            return False
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        # Check if the audio file is empty
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    return False


# Feature extraction function with four methods and only two hyperparameters
def extract_features(data_dir, audio_path, audio_name, n_mfcc, frame_size, hop_length):
    file_path = os.path.join(data_dir, audio_path, audio_name)
    # Check if the directory exists, if not, create it
    if os.path.exists(file_path):
        if os.path.getsize(file_path) <= 2048:
            # print(f"The audio file {file_path} is empty. Skipping this file.")
            return False
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=frame_size, hop_length=hop_length)
        sc = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
        sr = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_size, hop_length=hop_length)

        # Attempt to aggregate these features
        features = np.mean(np.vstack((mfccs, sc, sr, zcr)), axis=1)
        return features
    return False


# Dynamic Calculation for n_mels
def calculate_n_mels(segment_length, sample_rate, max_mels=40):
    # Ensure the number of Mel bands does not exceed half the segment length
    max_possible_mels = segment_length // 2
    return min(max_mels, max_possible_mels)


# Dynamic Calculation for frame_size
# def calculate_frame_size(segment_length, max_frame_size=1024):
#    return min(max_frame_size, 2**int(np.floor(np.log2(segment_length))))


# Adding noise if necessary to avoid nearly identical segments for kurtosis
def add_noise(data, noise_level=1e-10):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


# Making sure all features vectors are the same shape
def pad_or_truncate(features, target_length):
    if len(features) > target_length:
        return features[:target_length]
    elif len(features) < target_length:
        return np.pad(features, (0, target_length - len(features)), 'constant')
    else:
        return features


# Feature extraction function with all five methods and all three hyperparameters
def extract_features_with_segments(data_dir, audio_path, audio_name, n_mfcc, hop_length, frame_size, n_segments):
    file_path = os.path.join(data_dir, audio_path, audio_name)
    if os.path.exists(file_path):
        if os.path.getsize(file_path) <= 2048:
            return False

        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Handle case where n_segments is greater than audio length
        if n_segments > len(audio):
            n_segments = len(audio)

        segment_length = len(audio) // n_segments
        segment_features = []

        for i in range(n_segments):
            start_index = i * segment_length
            end_index = start_index + segment_length if (start_index + segment_length) <= len(audio) else len(audio)
            segment = audio[start_index:end_index]

            if len(segment) == 0:
                continue

            n_mels = calculate_n_mels(segment_length, sample_rate)
            # frame_size = calculate_frame_size(segment_length)

            mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc, n_fft=frame_size, hop_length=hop_length, n_mels=n_mels)
            sc = librosa.feature.spectral_centroid(y=segment, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
            sr = librosa.feature.spectral_rolloff(y=segment, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y=segment, frame_length=frame_size, hop_length=hop_length)

            # Compute kurtosis on the MFCCs
            if mfccs.shape[1] > 1:
                # Check for variance and add noise if necessary to avoid unreliable kurtosis because of nearly identical data
                if np.all(np.var(mfccs, axis=1) < 1e-10):
                    # print("Adding noise")
                    mfccs = add_noise(mfccs)
                kurtosis = scipy.stats.kurtosis(mfccs, axis=1, fisher=False)
                kurtosis_reshaped = np.tile(kurtosis[:, np.newaxis], (1, mfccs.shape[1]))
            else:
                kurtosis_reshaped = np.zeros_like(mfccs)

            # Aggregate features by computing the mean over time
            features = np.mean(np.vstack((mfccs, sc, sr, zcr, kurtosis_reshaped)), axis=1)
            segment_features.append(features)

        if segment_features:
            # Checking the most common shape for the feature vectors
            lengths = [len(f) for f in segment_features]
            # print(lengths)
            # most_common_length = mode(lengths).mode[0]

            # if there are different size vectors
            most_common_length_result = mode(lengths)
            if isinstance(most_common_length_result.mode, np.ndarray):
                most_common_length = most_common_length_result.mode[0]

                max_length = max(lengths)
                common_length_count = lengths.count(most_common_length)
                max_length_count = lengths.count(max_length)
                print(f"Most common length: {most_common_length}")
                print(f"Number of vectors with the most common length: {common_length_count}")
                print(f"Maximum length: {max_length}")
                print(f"Number of vectors with the maximum length: {max_length_count}")
            else:
                most_common_length = most_common_length_result.mode

            # Shaping all features vectors to use the most common length
            target_length = most_common_length
            segment_features = [pad_or_truncate(f, target_length) for f in segment_features]

            segment_features_flat = np.hstack(segment_features)
            return segment_features_flat
        else:
            return False
    return False
