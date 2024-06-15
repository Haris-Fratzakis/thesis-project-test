import pandas as pd
import numpy as np
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, Input
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
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
    # x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)  # random_state case

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
    logistic = LogisticRegression(solver='saga', max_iter=3000)

    print("LR Classifier Start")

    # Use GridSearchCV to search the hyperparameter grid with 5-fold cross validation
    #clf_1 = GridSearchCV(logistic, param_grid_l1, cv=5, verbose=0)
    clf_2 = GridSearchCV(logistic, param_grid_l2, cv=5, verbose=0, scoring='roc_auc', n_jobs=-1)

    # Fit the model with the grid search
    # clf_1.fit(x_train, y_train)
    clf_2.fit(x_train, y_train)

    # The clf.best_estimator_ now holds the model with the best combination of hyperparameters
    # if clf_1.best_score_ > clf_2.best_score_:
    #     model_lr = clf_1.best_estimator_
    # else:
    #     model_lr = clf_2.best_estimator_

    model_lr = clf_2.best_estimator_
    model_lr_hyper = clf_2.best_params_
    return model_lr, model_lr_hyper


# K-Nearest Neighbors Model
def knn_training(x_train, y_train, knn_hyper):
    param_grid = {
        'n_neighbors': list(range(knn_hyper[0][0], knn_hyper[0][1], knn_hyper[0][2])),
        'leaf_size': list(range(knn_hyper[1][0], knn_hyper[1][1], knn_hyper[1][2]))
    }
    # Create a KNN classifier instance
    knn = KNeighborsClassifier()

    print("kNN Classifier Start")

    # Create a GridSearchCV instance
    grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=0, scoring='roc_auc', n_jobs=-1)  # cv=5 for 5-fold cross-validation

    # Fit the GridSearchCV instance to the training data
    grid_search.fit(x_train, y_train)

    # Retrieve the best estimator (model with the best hyperparameters)
    model_knn = grid_search.best_estimator_
    model_knn_hyper = grid_search.best_params_

    return model_knn, model_knn_hyper


# Support Vector Machines Model
def svm_training(x_train, y_train, svm_hyper):
    param_grid = {
        'C': np.logspace(svm_hyper[0][0], svm_hyper[0][1], svm_hyper[0][2]),
        'gamma': np.logspace(svm_hyper[1][0], svm_hyper[1][1], svm_hyper[1][2]),
    }

    # Create an SVM classifier instance
    svm = SVC(probability=True, max_iter=3000)

    print("SVM Classifier Start")
    # Create a GridSearchCV instance
    grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=0, scoring='roc_auc', n_jobs=-1)  # cv=5 for 5-fold cross-validation

    # Fit the GridSearchCV instance to the training data
    grid_search.fit(x_train, y_train)

    # Retrieve the best estimator (model with the best hyperparameters)
    model_svm = grid_search.best_estimator_
    model_svm_hyper = grid_search.best_params_

    return model_svm, model_svm_hyper


# Multi-layer Perceptron Model
def mlp_training(x_train, y_train, mlp_hyper):
    # Define the parameter grid
    param_grid = {
        'hidden_layer_sizes': [(n,) for n in range(mlp_hyper[0][0], mlp_hyper[0][1], mlp_hyper[0][2])],  # (10,), (20,), ..., (100,)
        'alpha': [10 ** i for i in range(mlp_hyper[1][0], mlp_hyper[1][1])],  # 10^-7, 10^-6, ..., 10^7
        'learning_rate_init': np.arange(mlp_hyper[2][0], mlp_hyper[2][1], mlp_hyper[2][2])  # 0.05, 0.1, ..., 1
    }

    # Initialize model
    # model_mlp = Sequential([
    #     Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    #     Dense(64, activation='relu'),
    #     Dense(1, activation='sigmoid')  # Use 'softmax' and change units for multi-class
    # ])

    # Modify the MLPClassifier to catch and handle specific errors that occur due to non-finite weights:
    # class RobustMLPClassifier(MLPClassifier):
    #     def fit(self, x, y):
    #         try:
    #             super().fit(x, y)
    #         except ValueError as e:
    #             if 'non-finite' in str(e):
    #                 # print(f"Skipping non-finite weights error: {e}")
    #                 return None
    #             else:
    #                 raise

    # Define a function to create the Keras model
    def create_model(hidden_layer_sizes=(100,), learning_rate_init=0.001, alpha=0.0001, clip_norm=1.0):
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1],)))
        model.add(Dense(hidden_layer_sizes[0], activation='relu', kernel_regularizer=l2(alpha)))
        for units in hidden_layer_sizes[1:]:
            model.add(Dense(units, activation='relu', kernel_regularizer=l2(alpha)))
        model.add(Dense(1, activation='sigmoid'))  # Use 'softmax' and change units for multi-class classification

        optimizer = SGD(learning_rate=learning_rate_init, momentum=0.9, nesterov=True, clipnorm=clip_norm)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model


    # Create the MLPClassifier
    # mlp = RobustMLPClassifier(max_iter=1000, solver='sgd', early_stopping=True, n_iter_no_change=10)
    mlp = KerasClassifier(model=create_model, clip_norm=1.0, hidden_layer_sizes=(10,), alpha=1e-07, learning_rate_init=0.05, epochs=100, batch_size=32, verbose=0)

    print("MLP Classifier Start")

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, verbose=0, scoring='roc_auc', n_jobs=-1)

    # Fit the GridSearchCV instance to the training data
    grid_search.fit(x_train, y_train)

    # Retrieve the best estimator (model with the best hyperparameters)
    model_mlp = grid_search.best_estimator_
    model_mlp_hyper = grid_search.best_params_

    return model_mlp, model_mlp_hyper


# Convolutional Neural Network Model
def cnn_training(x_train, y_train, cnn_hyper):
    # Define the model creation function
    def create_model(num_filters=24, kernel_size=2, dropout_rate=0.1, dense_size=16):
        model = Sequential()
        model.add(Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), activation='relu',
                         input_shape=(64, 64, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))
        model.add(Flatten())
        model.add(Dense(dense_size, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(8, activation='relu'))  # Dense layer with 8 units
        model.add(Dense(2, activation='softmax'))  # Output layer for binary classification
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Wrap the Keras model with KerasClassifier
    # TODO Fix the CNN Classifier
    cnn = KerasClassifier(build_fn=create_model, verbose=0)

    # Define the parameter grid
    param_grid = {
        'num_filters': [3 * 2 ** k4 for k4 in range(cnn_hyper[0][0], cnn_hyper[0][1])],   # 3 × 2k4 where k4 = 3, 4, 5
        'kernel_size': range(cnn_hyper[1][0], cnn_hyper[1][1]),     # 2 and 3
        'dropout_rate': np.arange(cnn_hyper[2][0], cnn_hyper[2][1], cnn_hyper[2][2]),   # 0.1 to 0.5 in steps of 0.2
        'dense_size': [2 ** k5 for k5 in range(cnn_hyper[3][0], cnn_hyper[3][1])],   # 2k5 where k5 = 4, 5
        'batch_size': [2 ** k8 for k8 in range(cnn_hyper[4][0], cnn_hyper[4][1])],  # 2k8 where k8 = 6, 7, 8
        'epochs': range(cnn_hyper[5][0], cnn_hyper[5][1], cnn_hyper[5][2])  # 10 to 250 in steps of 20
    }

    print("CNN Classifier Start")

    # Define the early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=cnn, param_grid=param_grid, cv=5, verbose=0, scoring='roc_auc', n_jobs=-1)

    # Fit the GridSearchCV instance to the training data
    grid_search.fit(x_train, y_train, callbacks=[early_stopping])

    # Retrieve the best estimator (model with the best hyperparameters)
    model_cnn = grid_search.best_estimator_
    model_cnn_hyper = grid_search.best_params_

    return model_cnn, model_cnn_hyper


# Long Short-Term Memory Model
def lstm_training(x_train, y_train):
    # Define a function to create the model, required for KerasClassifier
    def create_model(dropout_rate, dense_size, lstm_units, learning_rate):
        model = Sequential()
        model.add(LSTM(lstm_units, activation='relu',  input_shape=(64, 64, 1), dropout=dropout_rate))
        model.add(Flatten())
        model.add(Dense(dense_size, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Create the KerasClassifier
    lstm = KerasClassifier(build_fn=create_model, verbose=0)

    # Define the parameter grid
    param_grid = {
        'dropout_rate': np.arange(0.1, 0.6, 0.2),   # 0.1 to 0.5 in steps of 0.2
        'dense_size': [2 ** k5 for k5 in [4, 5]],   # 2k5 where k5 = 4, 5
        'lstm_units': [2 ** k6 for k6 in [6, 7, 8]],    # 2k6 where k6 = 6, 7, 8
        'learning_rate': [10 ** k7 for k7 in [-2, -3, -4]],  # 10k7 where k7 = 􀀀 2, 􀀀 3, 􀀀 4
        'batch_size': [2 ** k8 for k8 in [6, 7, 8]],    # 2k8 where k8 = 6, 7, 8
        'epochs': np.arange(10, 251, 20)  # 10 to 250 in steps of 20
    }

    print("LSTM Classifier Start")

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=lstm, param_grid=param_grid, cv=5, verbose=0, scoring='roc_auc', n_jobs=-1)

    # Fit the GridSearchCV instance to the training data
    grid_search.fit(x_train, y_train)

    # Retrieve the best estimator (model with the best hyperparameters)
    model_lstm = grid_search.best_estimator_
    model_lstm_hyper = grid_search.best_params_

    return model_lstm, model_lstm_hyper


# Evaluate Performance of Classifiers
def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    # Binarize predictions based on threshold
    y_pred = [1 if prob > threshold else 0 for prob in y_pred_proba]

    # Calculating metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    precision = precision_score(y_true, y_pred, zero_division=0)
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
