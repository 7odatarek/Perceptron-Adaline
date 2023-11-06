import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron:
    def __init__(self, features, classes, lr, epochs, mse_threshold, bias):
        self.features = features
        self.mse_threshold = mse_threshold
        self.classes = classes
        self.lr = lr
        self.epochs = epochs
        self.weights = np.random.rand(len(features), 1)
        if (bias == True):
            self.bias = np.random.rand(1)
        else:
            self.bias = 0
        self.mse = 0

    def preprocessing(self, data):
        # choose specific data of two classes from column class
        for i in self.classes:
            data = data[data['Class'].isin(self.classes)]
        # convert classification column to numeric
        data['Class'] = pd.factorize(data['Class'])[0]
        # fill nan values with mean of the column
        data = data.fillna(data.mean())
        # scalling data
        data = (data - data.min()) / (data.max() - data.min())
        X = data[self.features]

        Y = data.iloc[:, -1]
        return X, Y

    def train_test_split(self, data):
        X, Y = self.preprocessing(data)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.4)

        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def signum(x):
        if x >= 0:
            return 1
        else:
            return 0

    def calc_mse(self, X, Y):
        Y = Y.reset_index(drop=True)
        mse = 0
        for i in range(len(Y)):
            mse += (Y[i] - np.dot(self.weights.T,
                    X.iloc[i]) + self.bias) ** 2
        return mse / len(Y)

    def train_model(self, X, Y):
        for i in range(self.epochs):
            self.mse = self.calc_mse(X, Y)
            if self.mse < self.mse_threshold:
                print('model is trained')
                break
            for j in range(len(X)):
                Y = Y.reset_index(drop=True)
                prediction = self.signum(
                    np.dot(self.weights.T, X.iloc[j]) + self.bias)
                error = Y[j] - prediction
                if (error == 0):
                    continue
                else:
                    self.weights[0] += self.lr * error * X.iloc[j, 0]
                    self.weights[1] += self.lr * error * X.iloc[j, 1]
                    self.bias += self.lr * error

    def draw_line(self, X):
        x = np.linspace(0, 1, 100)
        y = (-self.weights[0] * x - self.bias) / self.weights[1]
        plt.plot(x, y)
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1])
        plt.show()

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            predictions.append(self.signum(
                np.dot(self.weights.T, X.iloc[i]) + self.bias))
        return predictions

    def confusion_matrix(self, actual, predicted):
        actual = actual.reset_index(drop=True)
        predicted = pd.Series(predicted).reset_index(drop=True)
        # calculate confusion matrix
        cm = np.zeros((len(self.classes), len(self.classes)))
        for i in range(len(actual)):
            cm[int(actual[i])][int(predicted[i])] += 1
        # Convert to DataFrame and add labels
        cm_df = pd.DataFrame(cm, index=self.classes, columns=self.classes)
        return cm_df

    def calc_accuaracy(self, actual, predicted):
        # calculate accuracy
        acc = accuracy_score(actual, predicted)
        return acc
