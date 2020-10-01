import os
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # test 0.9449 accuracy cv = 10 (default)
import joblib


class SelectModelClass:
    X = None
    Y = None

    x_train, x_test, y_train, y_test = None, None, None, None

    def __init__(self):
        print("\nLoading model")


    def train_data_init_(self):
        print("Downloading MNIST dataset")
        mnist = fetch_openml('mnist_784', version=1, data_home=os.getcwd() + "\\data")
        print("Working on dataset")
        self.X, self.Y = mnist["data"], mnist["target"]
        self.Y = self.Y.astype(
            np.uint8)

        # self.X[self.X >= 50] = 255 # todo experiment
        # self.X[self.X < 50] = 0

        self.x_train, self.x_test, self.y_train, self.y_test = self.X[:60000], self.X[60000:], self.Y[:60000], self.Y[
                                                                                                               60000:]
    def manual_train_Knn(self):
        print("Training KNN model")
        self.train_data_init_()
        knn_clf = KNeighborsClassifier(n_neighbors=1)
        knn_clf.fit(self.X, self.Y)
        return knn_clf

    def load_model(self, name):
        if "binary" in name:
            print("Log : Using binary (0 or 255) trained model - " + name)
        elif "grayscale" in name:
            print("Log : Using grayscale (0 to 255) trained model - " + name)
        else:
            print("Log : Using saved model - " + name)

        model = joblib.load(os.getcwd() + "\\saved_model" + "\\" + name)
        return model

