import os
import librosa
import numpy as np
from scipy.stats import mode

# Feature Extraction Methods Used:
#   Mel Frequency Cepstral Coefficients(MFCCs)
#   Spectral Centroid (SC)
#   Spectral Roll-off(SR)
#   Zero-Crossing rate (ZCR)
#   Kurtosis

# Feature Extraction Methods based on Preliminary diagnosis of COVID-19 based on cough sounds using machine learning algorithms
# https://ieeexplore.ieee.org/abstract/document/9432324

# Feature Extraction Function with one method and one hyperparameter
def extract_features_simple(data_dir, audio_path, audio_name, n_mfcc):
    file_path = os.path.join(data_dir, audio_path, audio_name)
    # Check if the audio file is empty
    if os.path.exists(file_path):
        if os.path.getsize(file_path) <= 2048:
            # print(f"The audio file {file_path} is empty. Skipping this file.")
            return False
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    return False


# Feature Extraction Function with four methods and two hyperparameters
def extract_features(data_dir, audio_path, audio_name, n_mfcc, frame_size, hop_length):
    file_path = os.path.join(data_dir, audio_path, audio_name)
    # Check if the audio file is empty
    if os.path.exists(file_path):
        if os.path.getsize(file_path) <= 2048:
            # print(f"The audio file {file_path} is empty. Skipping this file.")
            return False
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=frame_size, hop_length=hop_length)
        sc = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
        sr = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_size, hop_length=hop_length)

        # Aggregate these features
        features = np.mean(np.vstack((mfccs, sc, sr, zcr)), axis=1)
        return features
    return False


# Function that dynamically calculates n_mels
def calculate_n_mels(segment_length, sample_rate, max_mels=40):
    # Ensure the number of Mel bands does not exceed half the segment length
    max_possible_mels = segment_length // 2
    return min(max_mels, max_possible_mels)


# Function that dynamically calculates frame_size
# def calculate_frame_size(segment_length, max_frame_size=1024):
#    return min(max_frame_size, 2**int(np.floor(np.log2(segment_length))))


# Function that adds noise if necessary to avoid nearly identical segments for kurtosis
def add_noise(data, noise_level=1e-5):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


# Function that makes sure all features vectors are the same shape
def pad_or_truncate(features, target_length):
    if len(features) > target_length:
        return features[:target_length]
    elif len(features) < target_length:
        return np.pad(features, (0, target_length - len(features)), 'constant')
    else:
        return features


# Feature extraction function with four methods and three hyperparameters
def extract_features_with_segments(data_dir, audio_path, audio_name, n_mfcc, hop_length, frame_size, n_segments):
    file_path = os.path.join(data_dir, audio_path, audio_name)
    # Check if the audio file is empty
    if os.path.exists(file_path):
        if os.path.getsize(file_path) <= 2048:
            print("Filepath with tiny size: ", file_path)
            return False

        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Handle case where n_segments is greater than audio length
        if n_segments > len(audio):
            print(len(audio))
            n_segments = len(audio)

        segment_length = len(audio) // n_segments
        segment_features = []
        low_variance_segment_counter = 0

        for i in range(n_segments):
            start_index = i * segment_length
            end_index = start_index + segment_length if (start_index + segment_length) <= len(audio) else len(audio)
            segment = audio[start_index:end_index]

            if len(segment) == 0:
                continue

            n_mels = calculate_n_mels(segment_length, sample_rate)
            # frame_size = calculate_frame_size(segment_length)

            mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc, n_fft=frame_size, hop_length=hop_length, n_mels=n_mels)
            # print("MFCCs in segment " + str(i))
            # print(mfccs)
            if np.isnan(mfccs).any():
                print("NaN Values for MFCCs for filepath: ", file_path)
                print("NaN MFCCs: ", mfccs)

            sc = librosa.feature.spectral_centroid(y=segment, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
            # print("SC in segment " + str(i))
            # print(sc)
            if np.isnan(sc).any():
                print("NaN Values for SC for filepath: ", file_path)
                print("NaN SC: ", sc)

            sr = librosa.feature.spectral_rolloff(y=segment, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
            # print("SR in segment " + str(i))
            # print(sr)
            if np.isnan(sr).any():
                print("NaN Values for SR for filepath: ", file_path)
                print("NaN SR: ", sr)

            zcr = librosa.feature.zero_crossing_rate(y=segment, frame_length=frame_size, hop_length=hop_length)
            # print("ZCR in segment " + str(i))
            # print(zcr)
            if np.isnan(zcr).any():
                print("NaN Values for ZCR for filepath: ", file_path)
                print("NaN ZCR: ", zcr)

            # Compute kurtosis on the MFCCs
            if mfccs.shape[1] > 1:
                # Check for variance and add noise if necessary to avoid unreliable kurtosis because of nearly identical data
                if not np.any(np.var(mfccs, axis=1) >= 1e-10):
                    if low_variance_segment_counter == 0:
                        pass
                    # print("Low variance sample: ", file_path)
                    # print("mfccs.shape[1]: ", mfccs.shape[1])
                    # print("k_segment: ", i)
                    # print("mfccs: ", mfccs)

                    low_variance_segment_counter += 1
                    pass
                    # print("Adding noise")
                    # mfccs = add_noise(mfccs, 1e-10)

                # kurtosis = scipy.stats.kurtosis(mfccs, axis=1, fisher=False)
                # kurtosis_reshaped = np.tile(kurtosis[:, np.newaxis], (1, mfccs.shape[1]))

            else:
                # print("mfccs.shape[1] = 0? : ", mfccs.shape[1])
                # kurtosis_reshaped = np.zeros_like(mfccs)
                pass

            # Aggregate features
            # Using Mean
            # features = np.mean(np.vstack((mfccs, sc, sr, zcr, kurtosis_reshaped)), axis=1)
            # features = np.mean(np.vstack((mfccs, sc, sr, zcr)), axis=1)

            # Without using Mean
            # features = np.vstack((mfccs, sc, sr, zcr, kurtosis_reshaped)).flatten()
            features = np.vstack((mfccs, sc, sr, zcr)).flatten()
            segment_features.append(features)

        if segment_features:
            # Check the most common shape for the feature vectors
            lengths = [len(f) for f in segment_features]
            # print(lengths)
            # most_common_length = mode(lengths).mode[0]

            # If there are different size vectors shape them all to use the most common length
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

            # Shape all features vectors to use the most common length
            target_length = most_common_length
            segment_features = [pad_or_truncate(f, target_length) for f in segment_features]

            segment_features_flat = np.hstack(segment_features)
            return segment_features_flat
        else:
            print("No segment features")
            return False
    return False
