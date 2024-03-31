import os
import librosa
import numpy as np
import scipy.stats


# Feature Extraction Methods Used:
#   Mel Frequency Cepstral Coefficients(MFCCs)
#   Spectral Centroid (SC)
#   Spectral Roll-off(SR)
#   Zero-Crossing rate (ZCR)
#   Kurtosis

# Feature Extraction Methods based on Preliminary diagnosis of COVID-19 based on cough sounds using machine learning algorithms
# https://ieeexplore.ieee.org/abstract/document/9432324

# Feature extraction function with only one method and only one hyperparameter
def extract_features_simple(data_dir, audio_paths_part_1, audio_paths_part_2,  n_mfcc):
    file_path = os.path.join(data_dir, audio_paths_part_1, audio_paths_part_2, "audio.cough.mp3")
    # Check if the directory exists, if not, create it
    if os.path.exists(file_path):
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    return False


# Feature extraction function with all five methods and only two hyperparameters
def extract_features(data_dir, audio_paths_part_1, audio_paths_part_2, n_mfcc, frame_size, hop_length):
    file_path = os.path.join(data_dir, audio_paths_part_1, audio_paths_part_2, "audio.cough.mp3")
    # Check if the directory exists, if not, create it
    if os.path.exists(file_path):
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=frame_size, hop_length=hop_length)
        sc = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
        sr = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_size, hop_length=hop_length)
        kurtosis = scipy.stats.kurtosis(mfccs, axis=1)
        kurtosis_reshaped = kurtosis[:, np.newaxis]  # Make it two-dimensional

        # Aggregate these features over the frames (currently just taking the mean across all frames)
        features = np.mean(np.vstack((mfccs, sc, sr, zcr, kurtosis_reshaped)), axis=1)
        return features
    return False


# Feature extraction function with all five methods and all three hyperparameters
def extract_features_with_segments(data_dir, audio_paths_part_1, audio_paths_part_2, n_mfcc, frame_size, hop_length, n_segments):
    file_path = os.path.join(data_dir, audio_paths_part_1, audio_paths_part_2, "audio.cough.mp3")
    # Check if the directory exists, if not, create it
    if os.path.exists(file_path):
        audio, sample_rate = librosa.load(file_path, sr=None)
        # Calculate the length of each segment
        segment_length = len(audio) // n_segments

        # Initialize a list to collect segment features
        segment_features = []

        for i in range(0, len(audio), segment_length):
            # Make sure we don't go past the end of the audio
            end_index = i + segment_length if (i + segment_length) <= len(audio) else len(audio)
            # Extract the segment
            segment = audio[i:end_index]

            # Extract features for this segment
            mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc, n_fft=frame_size, hop_length=hop_length)
            sc = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
            sr = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_size, hop_length=hop_length)
            kurtosis = scipy.stats.kurtosis(mfccs, axis=1)
            kurtosis_reshaped = kurtosis[:, np.newaxis]  # Make it two-dimensional

            # Aggregate these features over the frames (currently just taking the mean across all frames)
            features = np.mean(np.vstack((mfccs, sc, sr, zcr, kurtosis_reshaped)), axis=1)

            # Append the aggregated features of this segment to the list
            segment_features.append(features)

        # Combine all segment features into one flat vector per audio file
        segment_features_flat = np.hstack(segment_features)

        return segment_features_flat
    return False
