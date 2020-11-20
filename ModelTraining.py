import os
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # test 0.9449 accuracy cv = 10 (default)
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

import tensorflow as tf
from tensorflow import keras

class SelectModelClass:
    X = None
    Y = None

    x_train, x_test, y_train, y_test = None, None, None, None

    def __init__(self):
        print("\nLoading model")
        self.train_data_init_()

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

    def load_ensemble(self):
        default_save_path = os.getcwd() + "\\" + "Ensemble_KNN_ExTre_RandFor_grayscale"

        if os.path.exists(default_save_path): #100+ sec
            print("Loading saved model")
            voting_clf = joblib.load("Ensemble_KNN_ExTre_RandFor_grayscale")
            return voting_clf
        else: # 450 sec
            print("Training ensemble model")
            random_forest_clf = RandomForestClassifier(n_estimators = 100)
            extra_trees_clf = ExtraTreesClassifier(n_estimators = 100)
            knn_clf = KNeighborsClassifier(n_neighbors=1)

            classifiers = [("random_forest_clf", random_forest_clf),
                           ("extra_trees_clf", extra_trees_clf),
                           ("knn_clf", knn_clf),
                           ]

            voting_clf = VotingClassifier(classifiers)
            voting_clf.fit(self.X, self.Y)

            print("Saving model")
            joblib.dump(voting_clf,"Ensemble_KNN_ExTre_RandFor_grayscale")

            return voting_clf

    def train_ANN(self):
        print("Training Neural Network")
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape = [784]))
        model.add(keras.layers.Dense(100, activation = "relu"))
        model.add(keras.layers.Dense(100, activation = "relu"))
        model.add(keras.layers.Dense(10, activation = "softmax"))

        model.compile(loss="sparse_categorical_crossentropy",
             optimizer = "adam",
             metrics = ["accuracy"])

        model.fit(self.X,self.Y, epochs = 5)

        return model
