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
def extract_features_simple(data_dir, audio_path, audio_name, n_mfcc):
    file_path = os.path.join(data_dir, audio_path, audio_name)
    # Check if the directory exists, if not, create it
    if os.path.exists(file_path):
        if os.path.getsize(file_path) <= 2048:
            print(f"The audio file {file_path} is empty. Skipping this file.")
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
        # TODO fix the audio load just like it happens in extract_features_simple
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=frame_size, hop_length=hop_length)
        sc = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
        sr = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_size, hop_length=hop_length)

        # Attempt to aggregate these features
        features = np.mean(np.vstack((mfccs, sc, sr, zcr)), axis=1)
        return features
    return False


# Feature extraction function with all five methods and all three hyperparameters
def extract_features_with_segments(data_dir, audio_path, audio_name, n_mfcc, frame_size, hop_length, n_segments):
    # TODO FIX THIS
    file_path = os.path.join(data_dir, audio_path, audio_name)
    if os.path.exists(file_path):
        audio, sample_rate = librosa.load(file_path, sr=None)
        segment_length = len(audio) // n_segments
        segment_features = []

        for i in range(0, len(audio), segment_length):
            end_index = i + segment_length if (i + segment_length) <= len(audio) else len(audio)
            segment = audio[i:end_index]

            mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc, n_fft=frame_size,
                                         hop_length=hop_length)
            sc = librosa.feature.spectral_centroid(y=segment, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
            sr = librosa.feature.spectral_rolloff(y=segment, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y=segment, frame_length=frame_size, hop_length=hop_length)
            kurtosis = scipy.stats.kurtosis(mfccs, axis=1)
            kurtosis_reshaped = np.tile(kurtosis[:, np.newaxis], (1, mfccs.shape[1]))

            features = np.mean(np.vstack((mfccs, sc, sr, zcr, kurtosis_reshaped)), axis=1)
            segment_features.append(features)

        segment_features_flat = np.hstack(segment_features)
        return segment_features_flat
    return False
