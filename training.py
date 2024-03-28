import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling2D, LSTM
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Classifiers
#   Logistic Regression (LR)
#   K-Nearest Neighbor (KNN)
#   Support Vector Machines (SVM)
#   Multi-layer Perceptron (MLP)
#   Convolutional Neural Network (CNN)
#   Long Short-Term Memory (LSTM)

# Classifiers based on COVID-19 cough classification using machine learning and global smartphone recordings
# https://www.sciencedirect.com/science/article/pii/S0010482521003668


# Test Function for Classifiers
def classifier(features, labels, n_mfcc, frame_size=0, n_segments=0):
    # Splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Initialize all models
    model_lr = lr_training(x_train, y_train)
    model_knn = knn_training(x_train, y_train)
    model_svm = svm_training(x_train, y_train)
    model_mlp = mlp_training(x_train, y_train)
    model_cnn = cnn_training(x_train, y_train)
    model_lstm = lstm_training(x_train, y_train)

    # Predict and evaluate all models
    # LR
    y_pred_proba = model_lr.predict_proba(x_test)[:, 1]
    performance_metrics_lr = evaluate_model(y_test, y_pred_proba)

    # KNN
    y_pred_proba = model_knn.predict_proba(x_test)[:, 1]
    performance_metrics_knn = evaluate_model(y_test, y_pred_proba)

    # SVM
    y_pred_proba = model_svm.predict_proba(x_test)[:, 1]
    performance_metrics_svm = evaluate_model(y_test, y_pred_proba)

    # MLP
    y_pred_proba = model_mlp.predict(x_test).ravel()
    performance_metrics_mlp = evaluate_model(y_test, y_pred_proba)

    # CNN
    y_pred_proba = model_cnn.predict(x_test).ravel()
    performance_metrics_cnn = evaluate_model(y_test, y_pred_proba)

    # LSTM
    y_pred_proba = model_lstm.predict(x_test).ravel()
    performance_metrics_lstm = evaluate_model(y_test, y_pred_proba)

    # Save results
    results = {
        'mfcc': n_mfcc,
        'frame_size': frame_size,
        'segments': n_segments,
        'performance_metrics_lr': performance_metrics_lr,
        'performance_metrics_knn': performance_metrics_knn,
        'performance_metrics_svm': performance_metrics_svm,
        'performance_metrics_mlp': performance_metrics_mlp,
        'performance_metrics_cnn': performance_metrics_cnn,
        'performance_metrics_lstm': performance_metrics_lstm,
    }

    return results


# Logistic Regression Model
def lr_training(x_train, y_train):
    # Initialize model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    return model


# K-Nearest Neighbors Model
def knn_training(x_train, y_train):
    # Initialize model
    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(x_train, y_train)

    return model_knn


# Support Vector Machines Model
def svm_training(x_train, y_train):
    # Initialize model
    model_svm = SVC(kernel='linear')  # You can try different kernels like 'rbf'
    model_svm.fit(x_train, y_train)

    return model_svm


# Multi-layer Perceptron Model
def mlp_training(x_train, y_train):
    # Initialize model
    model_mlp = Sequential([
        Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Use 'softmax' and change units for multi-class
    ])

    model_mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_mlp.fit(x_train, y_train, epochs=10, batch_size=32)

    return model_mlp


# Convolutional Neural Network Model
def cnn_training(x_train, y_train):
    # Initialize model
    model_cnn = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_cnn.fit(x_train, y_train, epochs=10, batch_size=32)

    return model_cnn


# Long Short-Term Memory Model
def lstm_training(x_train, y_train):
    # Initialize model
    model_lstm = Sequential([
        LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2])),
        Dense(1, activation='sigmoid')
    ])

    model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_lstm.fit(x_train, y_train, epochs=10, batch_size=32)

    return model_lstm


# Evaluate Performance of Classifiers
def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    # Binarize predictions based on threshold
    y_pred = [1 if prob > threshold else 0 for prob in y_pred_proba]

    # Calculating metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)  # AUC requires probability scores of the positive class

    return [specificity, sensitivity, precision, accuracy, f1, auc]
