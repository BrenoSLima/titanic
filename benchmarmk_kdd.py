#--Modules--
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time

#--ML Models--
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier

#--Tools--
from sklearn.model_selection import train_test_split
from titanic_preprocessing import data_cleaner

def NearestNeighbors(X_train, X_test, Y_train, Y_test):
    knn_settings = range(1, 50, 5)
    training_accuracy = []
    test_accuracy = []
    
    for n_neighbors in knn_settings:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, Y_train)
        training_accuracy.append(knn.score(X_train, Y_train))
        test_accuracy.append(knn.score(X_test, Y_test))
    
    plot_model(knn_settings, training_accuracy, test_accuracy, "n_neighbors", "Nearest Neighbor")
    
   
def LinearRidgeRegression(X_train, X_test, Y_train, Y_test):
    ridge_settings = range(1, 2000, 200)
    training_accuracy = []
    test_accuracy = []
    
    for alpha in ridge_settings:
        ridge = RidgeClassifier(alpha=alpha)
        ridge.fit(X_train, Y_train)
        training_accuracy.append(ridge.score(X_train, Y_train))
        test_accuracy.append(ridge.score(X_test, Y_test))
    
    plot_model(ridge_settings, training_accuracy, test_accuracy, "alpha", "Ridge Regression")

def DecisionTree(X_train, X_test, Y_train, Y_test):    
    tree_settings = range(1, 20)
    training_accuracy = []
    test_accuracy = []
    
    for max_depth in tree_settings:
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(X_train, Y_train)
        training_accuracy.append(tree.score(X_train, Y_train))
        test_accuracy.append(tree.score(X_test, Y_test))
    
    plot_model(tree_settings, training_accuracy, test_accuracy, "max_depth", "Decision Tree")

def plot_model(X, Y_train, Y_test, xlabel, title):
    plt.figure()
    plt.plot(X, Y_train, label="Training accuracy")
    plt.plot(X, Y_test, label="Test accuracy")
    plt.ylabel("Acuracy")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()

def benchmark(X_train, X_test, Y_train, Y_test):
    ml_models = [NearestNeighbors,
                 LinearRidgeRegression,
                 DecisionTree]

    for model_function in ml_models:
        model_function(X_train, X_test, Y_train, Y_test)


X, y = data_cleaner('Desktop/Python/titanic/data/', 'train.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y)

benchmark(X_train, X_test, y_train, y_test)

