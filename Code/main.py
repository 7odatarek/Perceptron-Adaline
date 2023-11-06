from Perceptron import Perceptron
from Adaline import Adaline
import pandas as pd

def workFlow(feature1,feature2,class1,class2,learning_rate,epochs,mse_threshold,add_bias,algorithm):
    data = pd.read_excel('Dry_Bean_Dataset.xlsx')
    features = [feature1, feature2]
    classes = [class1, class2]
    lr = learning_rate
    epochs = epochs
    mse_threshold = mse_threshold
    if add_bias:
        bias = True
    else:
        bias = False

    if algorithm == "Perceptron":
        p = Perceptron(features, classes, lr, epochs, mse_threshold, bias)
        X, Y = p.preprocessing(data)
        X_train, X_test, Y_train, Y_test = p.train_test_split(data)
        p.train_model(X_train, Y_train)
        p.draw_line(X_train)
        prediction = p.predict(X_test)
        cm = p.confusion_matrix(Y_test, prediction)
        acc = p.calc_accuaracy(Y_test, prediction)
        print("Accuracyacc = ", round(acc * 100, 2), "%")
        print("Confution Matrix:")
        print(cm)
        print("///////////////////////////////////////////////")
    elif algorithm == "Adaline":
        a = Adaline(features, classes, lr, epochs, mse_threshold, bias)
        X, Y = a.preprocessing(data)
        X_train, X_test, Y_train, Y_test = a.train_test_split(data)
        a.train_model(X_train, Y_train)
        a.draw_line(X_train)
        predictions = a.predict(X_test)
        cm = a.confusion_matrix(Y_test, predictions)
        acc = a.calc_accuracy(Y_test, predictions)
        print("Accuracy = ", round(acc * 100, 2), "%")
        print("Confution Matrix:")
        print(cm)
        print(predictions)
