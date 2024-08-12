import numpy as np
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, Input
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam, SGD
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


# Function for Logistic Regression Model
def lr_training(x_train, y_train, lr_hyper=None, random_state=None):
    if lr_hyper is None:
        lr_hyper = [[-1]]

    # Random_state case
    if random_state is not None:
        np.random.seed(random_state)

    # create the hyperparameter grid for each penalty (l2 only)
    param_grid_l2 = {
        'penalty': ['l2']
    }

    if lr_hyper[0] != [-1]:
        param_grid_l2['C'] = np.logspace(lr_hyper[0][0], lr_hyper[0][1], lr_hyper[0][2])

    # Create a logistic regression model
    logistic = LogisticRegression(solver='saga', max_iter=6000, random_state=random_state)

    print("LR Classifier Start")

    # Use GridSearchCV to search the hyperparameter grid with 5-fold cross validation
    clf_2 = GridSearchCV(logistic, param_grid_l2, cv=5, verbose=0, scoring='roc_auc', n_jobs=-1)

    # Fit the model with the grid search
    clf_2.fit(x_train, y_train)

    model_lr = clf_2.best_estimator_
    model_lr_hyper = clf_2.best_params_
    return model_lr, model_lr_hyper


# Function for K-Nearest Neighbors Model
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


# Support Vector Machine Model
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


# Function for Multi-layer Perceptron Model
def mlp_training(x_train, y_train, mlp_hyper):
    # Define the parameter grid
    param_grid = {
        'hidden_layer_sizes': [(n,) for n in range(mlp_hyper[0][0], mlp_hyper[0][1], mlp_hyper[0][2])],  # (10,), (20,), ..., (100,)
        'alpha': [10 ** i for i in range(mlp_hyper[1][0], mlp_hyper[1][1])],  # 10^-7, 10^-6, ..., 10^7
        'learning_rate_init': np.arange(mlp_hyper[2][0], mlp_hyper[2][1], mlp_hyper[2][2])  # 0.05, 0.1, ..., 1
    }

    # Create the MLPClassifier
    mlp = MLPClassifier(solver='adam', early_stopping=True, n_iter_no_change=10, max_iter=200)

    print("MLP Classifier Start")

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, verbose=0, scoring='roc_auc', n_jobs=-1)

    # Fit the GridSearchCV instance to the training data
    grid_search.fit(x_train, y_train)

    # Retrieve the best estimator (model with the best hyperparameters)
    model_mlp = grid_search.best_estimator_
    model_mlp_hyper = grid_search.best_params_

    return model_mlp, model_mlp_hyper


# Function for Convolutional Neural Network Model
# NOT CURRENTLY BEING USED
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


# Function for Long Short-Term Memory Model
# NOT CURRENTLY BEING USED
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


# Function to evaluate the performance of the classifiers using probabilities
def evaluate_pred_proba_model(y_true, y_pred_proba, threshold=0.5):
    # Binarize predictions based on threshold
    y_pred = [1 if prob > threshold else 0 for prob in y_pred_proba]

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    precision = precision_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)

    # roc_curve_plot(y_true, y_pred_proba)

    return [specificity, sensitivity, precision, accuracy, f1, auc]


# Function to evaluate the performance of the classifiers for Ensemble Learning with Hard Voting
def evaluate_pred_model(y_true, y_pred):
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    precision = precision_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return [specificity, sensitivity, precision, accuracy, f1]


# Function to plot the ROC curve
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
