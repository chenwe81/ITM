"""
This script performs class split simulation using different classifiers and evaluates 
their performance using various metrics and crossvalidation schemes.
"""
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

n = 100       # number of samples
dim = 15      # dimension
CVfolds = 20  # number of cross-validation folds

classifiers = ['SVMlinear', 'SVMrbf', 'Perceptron']

for clf in classifiers:
    start_time = time.time()
    LOO_mcc = np.zeros([n, CVfolds])
    LOO_err = np.zeros([n, CVfolds])
    LOO_auc = np.zeros([n, CVfolds])
    LOO_roc = np.zeros([n, CVfolds])
    LOO_acc = np.zeros([n, CVfolds])
    SKF_mcc = np.zeros([n, CVfolds])
    SKF_err = np.zeros([n, CVfolds])
    SKF_auc = np.zeros([n, CVfolds])
    SKF_roc = np.zeros([n, CVfolds])
    SKF_acc = np.zeros([n, CVfolds])

    for cv in range(CVfolds):
        print(clf + f" cross-validation fold {cv+1}/{CVfolds}")
        for nn in range(2, n-1):

            # create a random dataset with n samples and dim features
            # the target variable y is binary and balancing varies from one side to the other
            X = np.random.rand(n, dim)
            y = np.concatenate((np.ones(n-nn), np.zeros(nn)))

            # Create the classifier
            if clf == 'SVMlinear':
                classifier = SVC(kernel="linear")
            elif clf == 'SVMrbf':
                classifier = SVC(kernel="rbf")
            elif clf == 'Perceptron':
                classifier = Perceptron()

            # Leave-One-Out cross-validation
            loo = LeaveOneOut()
            predictions = []
            for i, (train_index, test_index) in enumerate(loo.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                predictions.append(y_pred[0])

            LOO_mcc[nn,cv] = matthews_corrcoef(y, predictions)
            LOO_err[nn,cv] = 1 - accuracy_score(y, predictions)
            LOO_acc[nn,cv] = accuracy_score(y,predictions)
            LOO_auc[nn,cv] = auc(y,predictions)
            LOO_roc[nn,cv] = roc_auc_score(y,predictions)

            # Stratified K-Fold cross-validation
            warnings.filterwarnings("ignore", category=UserWarning)
            skf = StratifiedKFold(n_splits=10, shuffle=False)
            predictions = np.zeros(len(y))
            for i, (train_index, test_index) in enumerate(skf.split(X,y)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                predictions[test_index] = y_pred

            SKF_mcc[nn,cv] = matthews_corrcoef(y, predictions)
            SKF_err[nn,cv] = 1 - accuracy_score(y, predictions)
            SKF_acc[nn,cv] = accuracy_score(y,predictions)
            SKF_auc[nn,cv] = auc(y,predictions)
            SKF_roc[nn,cv] = roc_auc_score(y,predictions)

    # Plot LOO
    mean_LOO_acc = np.mean(LOO_err, axis=1)
    plt.plot(mean_LOO_acc[2:98], label='LOO Error')

    mean_LOO_acc = np.mean(LOO_acc, axis=1)
    plt.plot(mean_LOO_acc[2:98], label='LOO Accuracy')

    mean_LOO_auc = np.mean(LOO_auc, axis=1)
    plt.plot(mean_LOO_auc[2:98], label='LOO AUC')

    mean_LOO_roc = np.mean(LOO_roc, axis=1)
    plt.plot(mean_LOO_roc[2:98], label='LOO ROC AUC')

    mean_LOO_mcc = np.mean(LOO_mcc, axis=1)
    plt.plot(mean_LOO_mcc[2:98], label='LOO MCC')

    # Add labels and legend
    plt.xlabel('Class Label Split')
    plt.ylabel('Score')
    plt.xticks(np.arange(10, 91, 10))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(clf)
    plt.show()

    # Clear the plot
    plt.clf()

    # Plot SKF
    mean_SKF_acc = np.mean(SKF_err, axis=1)
    plt.plot(mean_SKF_acc[2:98], label='SKF Error')

    mean_SKF_acc = np.mean(SKF_acc, axis=1)
    plt.plot(mean_SKF_acc[2:98], label='SKF Accuracy')

    mean_SKF_auc = np.mean(SKF_auc, axis=1)
    plt.plot(mean_SKF_auc[2:98], label='SKF AUC')

    mean_SKF_roc = np.mean(SKF_roc, axis=1)
    plt.plot(mean_SKF_roc[2:98], label='SKF ROC AUC')

    mean_SKF_mcc = np.mean(SKF_mcc, axis=1)
    plt.plot(mean_SKF_mcc[2:98], label='SKF MCC')

    # Add labels and legend
    plt.xlabel('Class Label Split')
    plt.ylabel('Score')
    plt.xticks(np.arange(10, 91, 10))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(clf)
    plt.show()

    # Clear the plot
    plt.clf()

    end_time = time.time()
    print(f"Time elapsed for {clf}: {round((end_time - start_time)/60, 2)} minutes")
    print("")
