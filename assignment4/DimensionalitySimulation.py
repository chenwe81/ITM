"""
This script performs dimensionality simulation by generating random datasets 
with varying dimensions and evaluating the performance of different 
classifiers using cross-validation.

The script imports the necessary libraries: 
numpy, time, matplotlib.pyplot, sklearn.svm.SVC, sklearn.linear_model.Perceptron, 
sklearn.model_selection.LeaveOneOut, sklearn.model_selection.StratifiedKFold, 
sklearn.metrics.matthews_corrcoef, and sklearn.metrics.accuracy_score.

The script defines the following variables:
- n: number of samples
- maxdim: maximum dimension
- CVfolds: number of cross-validation folds

The script defines a list of classifiers: ['SVMlinear', 'SVMrbf', 'Perceptron'].

For each classifier, the script performs the following steps:
1. Initializes arrays to store the results of Leave-One-Out (LOO) 
   and Stratified K-Fold (SKF) cross-validation.
2. Iterates over the specified number of cross-validation folds.
3. Prints the current cross-validation fold.
4. Iterates over the range of dimensions.
5. Generates a random dataset with the specified number of samples and dimension.
6. Creates the classifier based on the current classifier type.
7. Performs Leave-One-Out cross-validation and stores the predictions.
8. Calculates the Matthews Correlation Coefficient and accuracy for LOO.
9. Performs Stratified K-Fold cross-validation and stores the predictions.
10. Calculates the Matthews Correlation Coefficient and accuracy for SKF.
11. Plots the LOO error, LOO MCC, SKF error, and SKF MCC.
12. Saves the plot as a PNG image.
13. Prints the elapsed time for the current classifier.

Note: The script assumes that the necessary libraries are installed and the file paths are correct.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

n = 10          # number of samples
maxdim = 10000  # maximum dimension
CVfolds = 100   # number of cross-validation folds

classifiers = ['SVMlinear', 'SVMrbf', 'Perceptron']

for clf in classifiers:
    start_time = time.time()
    LOO_mcc = np.zeros([maxdim, CVfolds])
    LOO_err = np.zeros([maxdim, CVfolds])
    SKF_mcc = np.zeros([maxdim, CVfolds])
    SKF_err = np.zeros([maxdim, CVfolds])

    for cv in range(CVfolds):
        print(clf + f" cross-validation fold {cv+1}/{CVfolds}")
        for dim in range(1, maxdim):

            # create a random dataset with n samples and dim features
            # the target variable y is binary and balanced
            X = np.random.rand(n, dim)
            y = np.concatenate((np.ones(n//2), np.zeros(n//2)))

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
                #print(f"Fold {i}:")
                #print(f"  Train: index={train_index}")
                #print(f"  Test:  index={test_index}")
            LOO_mcc[dim,cv] = matthews_corrcoef(y, predictions)
            LOO_err[dim,cv] = 1 - accuracy_score(y, predictions)
            #print("Matthews Correlation Coefficient:", mcc)
            #print("Accuracy:", acc)

            # Stratified K-Fold cross-validation
            skf = StratifiedKFold(n_splits=n//2, shuffle=False)
            predictions = np.zeros(len(y))
            for i, (train_index, test_index) in enumerate(skf.split(X,y)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                predictions[test_index] = y_pred
                #print(f"Fold {i}:")
                #print(f"  Train: index={train_index}")
                #print(f"  Test:  index={test_index}")
            SKF_mcc[dim,cv] = matthews_corrcoef(y, predictions)
            SKF_err[dim,cv] = 1 - accuracy_score(y, predictions)
            #print("Matthews Correlation Coefficient:", mcc)
            #print("Accuracy:", acc)

    # Plot LOO_acc
    mean_LOO_acc = np.mean(LOO_err, axis=1)
    plt.plot(mean_LOO_acc, label='LOO Error')

    # Plot LOO_mcc range(3, maxdim),
    mean_LOO_mcc = np.mean(LOO_mcc, axis=1)
    plt.plot(mean_LOO_mcc, label='LOO MCC')

    # Plot SKF_acc
    mean_SKF_acc = np.mean(SKF_err, axis=1)
    plt.plot(mean_SKF_acc, label='SKF Error')

    # Plot SKF_mcc
    mean_SKF_mcc = np.mean(SKF_mcc, axis=1)
    plt.plot(mean_SKF_mcc, label='SKF MCC')

    # Add labels and legend
    plt.xlabel('Dimension')
    plt.ylabel('Score')
    plt.legend()

    # Set x-axis to log scale
    plt.xscale('log')

    # Save the plot as a PNG image
    plt.title(clf)
    plt.savefig('DimensionalitySimulation-' + clf + '.png')

    # Clear the plot
    plt.clf()

    end_time = time.time()
    print(f"Time elapsed for {clf}: {round((end_time - start_time)/60, 2)} minutes")
    print("")
