import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling2D, LSTM
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    roc_curve, auc
import matplotlib.pyplot as plt


# Classifiers
#   Logistic Regression (LR)
#   K-Nearest Neighbor (KNN)
#   Support Vector Machines (SVM)
#   Multi-layer Perceptron (MLP)
#   Convolutional Neural Network (CNN)
#   Long Short-Term Memory (LSTM)

# Classifiers and Hyperparameters based on COVID-19 cough classification using machine learning and global smartphone recordings
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
        'performance_metrics_lstm': performance_metrics_lstm
    }

    return results


# Logistic Regression Model
def lr_training(x_train, y_train, lr_hyper=None):
    if lr_hyper is None:
        lr_hyper = [[-1]]

    # create the hyperparameter grid for each penalty (l1 and l2)
    param_grid_l1 = {
        'penalty': ['l1']
    }

    param_grid_l2 = {
        'penalty': ['l2']
    }

    if lr_hyper[0] != [-1]:
        # param_grid_l1['C'] = np.logspace(lr_hyper[0][0], lr_hyper[0][1], lr_hyper[0][2])
        param_grid_l2['C'] = np.logspace(lr_hyper[0][0], lr_hyper[0][1], lr_hyper[0][2])

    # Create a logistic regression model
    logistic = LogisticRegression(solver='saga', max_iter=10000)

    print("LR Classifier Start")

    # Use GridSearchCV to search the hyperparameter grid with 5-fold cross validation
    #clf_1 = GridSearchCV(logistic, param_grid_l1, cv=5, verbose=0)
    clf_2 = GridSearchCV(logistic, param_grid_l2, cv=5, verbose=0, scoring='roc_auc')

    # Fit the model with the grid search
    # clf_1.fit(x_train, y_train)
    clf_2.fit(x_train, y_train)

    # The clf.best_estimator_ now holds the model with the best combination of hyperparameters
    # if clf_1.best_score_ > clf_2.best_score_:
    #     model_lr = clf_1.best_estimator_
    # else:
    #     model_lr = clf_2.best_estimator_

    model_lr = clf_2.best_estimator_

    return model_lr


# K-Nearest Neighbors Model
def knn_training(x_train, y_train, knn_hyper):
    param_grid = {
        'n_neighbors': list(range(knn_hyper[0][0], knn_hyper[0][1], knn_hyper[0][1])),
        'leaf_size': list(range(knn_hyper[1][0], knn_hyper[1][1], knn_hyper[1][2]))
    }

    # Create a KNN classifier instance
    knn = KNeighborsClassifier()

    print("kNN Classifier Start")

    # Create a GridSearchCV instance
    grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=0, scoring='roc_auc')  # cv=5 for 5-fold cross-validation

    # Fit the GridSearchCV instance to the training data
    grid_search.fit(x_train, y_train)

    # Retrieve the best estimator (model with the best hyperparameters)
    model_knn = grid_search.best_estimator_

    return model_knn


# Support Vector Machines Model
def svm_training(x_train, y_train, svm_hyper):
    param_grid = {
        'C': np.logspace(svm_hyper[0][0], svm_hyper[0][1], svm_hyper[0][2]),
        'gamma': np.logspace(svm_hyper[1][0], svm_hyper[1][1], svm_hyper[1][2]),
    }

    # Create an SVM classifier instance
    svm = SVC()

    print("SVM Classifier Start")
    # Create a GridSearchCV instance
    grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=0, scoring='roc_auc')  # cv=5 for 5-fold cross-validation

    # Fit the GridSearchCV instance to the training data
    grid_search.fit(x_train, y_train)

    # Retrieve the best estimator (model with the best hyperparameters)
    model_svm = grid_search.best_estimator_

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

    # roc_curve_plot(y_true, y_pred_proba)

    return [specificity, sensitivity, precision, accuracy, f1, auc]


# Plot the ROC curve to investigate the metrics
def roc_curve_plot(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
