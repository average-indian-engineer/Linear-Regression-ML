import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import copy
from random import randrange
from sklearn import preprocessing
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
import pickle

scaleElement = lambda x, min, max : (x-min)/(max-min)

def readCleanIncomeData(filename):
    headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
    dataset = pd.read_csv(filename, header=None, names=headers)
    obj_df = dataset.select_dtypes(include=['object'])
    for key in obj_df.keys():
        dataset[key] = dataset[key].astype('category')
        dataset[key] = dataset[key].cat.codes
    return dataset.values


def generateSplits(dataset, folds=5):
    datasetCopy = list(dataset)
    datasetSplit = []
    foldSize = len(dataset) // folds
    for i in range(folds):
        fold = []
        while foldSize > len(fold):
            j = randrange(len(datasetCopy))
            fold.append(datasetCopy.pop(j))
        datasetSplit.append(fold)
    return np.array(datasetSplit)

def seprateInpOut(dataset):
    X = dataset[:,:-1]
    y = dataset[:,-1]
    y = y.reshape((y.shape[0], 1))
    return X, y

def sigmoid(z):
    return .5 * (1 + np.tanh(.5 * z))

# Hypothesis
def hypothesis(theta, X):
    z = np.dot(X, theta)
    return sigmoid(z)
    
def getRmseCost(predict, actual):
    m = predict.shape[0]
    return (sum(np.square(np.subtract(predict, actual))) / m) ** 0.5

def gradientDescent(X, y, n, alpha, nIterations, l1=None, l2=None):
    cost = np.zeros(nIterations)
    accuracy = np.zeros(nIterations)
    m = X.shape[0]
    theta = np.zeros((n, 1))
    h = hypothesis(theta, X)
    for i in range(nIterations):
        oldTheta = copy.deepcopy(theta)
        oldTheta[0] = 0
        if(l1):
            theta = theta - (alpha/m) * np.transpose(np.matmul(np.transpose(h - y), X)) - (l1 * alpha / m) / 2 * signBits(oldTheta)
        elif(l2):
            theta = theta - (alpha/m) * np.transpose(np.matmul(np.transpose(h - y), X)) - (l2 * alpha / m) * oldTheta  
        else:
            theta = theta - (alpha/m) * np.transpose(np.matmul(np.transpose(h - y), X))
        h = hypothesis(theta, X)
        cost[i] = getRmseCost(h, y)
        accuracy[i] = (hypothesis(theta, X).round() == y).mean()
    return theta, cost, accuracy


def minMaxScale(X):
    m, n = X.shape
    for j in range(1, n):
        max = -float('inf')
        min = float('inf')
        for i in range(m):
            curE = X[i][j]
            if(curE > max):
                max = curE
            if(curE < min):
                min = curE
        if(min == max):
            X[:,j] = [0 * m]
        else:
            X[:,j] = [scaleElement(e, min, max) for e in X[:,j]]            
    return X


def signBits(X):
    signArr = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        e = X[i]
        val = 0
        if(e > 0):
            val = 1
        elif(e < 0):
            val = -1
        signArr[i] = val
    return signArr

def getAccuracy(X, y, theta, thetaL1, thetaL2):
    acc = (hypothesis(theta, X).round() == y).mean() * 100
    accL1 = (hypothesis(thetaL1, X).round() == y).mean() * 100
    accL2 = (hypothesis(thetaL2, X).round() == y).mean() * 100
    return str(round(acc, 7)), str(round(accL1, 7)), str(round(accL2, 7))

def saveModels():
    pickle.dump(theta, open('./q2amodels/theta.sav', 'wb'))
    pickle.dump(rmse, open('./q2amodels/rmse.sav', 'wb'))
    pickle.dump(accuracy, open('./q2amodels/accuracy.sav', 'wb'))

    pickle.dump(L2, open('./q2amodels/L2.sav', 'wb'))    
    pickle.dump(thetaL2, open('./q2amodels/thetaL2.sav', 'wb'))    
    pickle.dump(rmseL2, open('./q2amodels/rmseL2.sav', 'wb'))
    pickle.dump(accuracyL2, open('./q2amodels/accuracyL2.sav', 'wb'))

    pickle.dump(L1, open('./q2amodels/L1.sav', 'wb'))    
    pickle.dump(thetaL1, open('./q2amodels/thetaL1.sav', 'wb'))    
    pickle.dump(rmseL1, open('./q2amodels/rmseL1.sav', 'wb'))
    pickle.dump(accuracyL1, open('./q2amodels/accuracyL1.sav', 'wb'))

def logisticRegressionLoader(useSavedModels):
    print("-" * 10 + "" + "-" * 10)    
    print("Collecting Data...")
    print("Encoding Categories...")
    incomeData = readCleanIncomeData('./q2dataset/income_train.csv')
    incomeData_test = readCleanIncomeData('./q2dataset/income_test.csv')

    splitData = generateSplits(incomeData)
    incomeData_val = splitData[0]
    print("Generating Splits...")
    incomeData_train = []
    for group in splitData[1:]:
        for e in group:
            incomeData_train.append(e)
    incomeData_train = np.array(incomeData_train)
    
    X_train, y_train = seprateInpOut(incomeData_train)
    X_test, y_test = seprateInpOut(incomeData_test)
    X_val, y_val = seprateInpOut(incomeData_val)

    oneCol = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((oneCol, X_train), axis=1)
    oneCol = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((oneCol, X_test), axis=1)
    oneCol = np.ones((X_val.shape[0], 1))
    X_val = np.concatenate((oneCol, X_val), axis=1)

    n = X_train.shape[1]
    alpha = 1.11
    nIterations = 1000
    X_train = minMaxScale(X_train)
    theta = None
    rmse = None
    accuracy = None
    L2 = None
    thetaL2 = None
    rmseL2 = None
    accuracyL2 = None
    L1 = None
    thetaL1 = None
    rmseL1 = None
    accuracyL1 = None

    if(not useSavedModels):
        print("Logistic Regression...")
        theta, rmse, accuracy = gradientDescent(X_train, y_train, n, alpha, nIterations)
        
        print("Running L2, Ridge Regression...")
        params = {'alpha': np.linspace(0.5, 1.0, num=100)}
        rdg_reg = Ridge()
        clf = GridSearchCV(rdg_reg, params, cv=5, scoring = 'neg_mean_squared_error')
        clf.fit(X_train, y_train)
        L2 = round(clf.best_params_['alpha'], 5)
        thetaL2, rmseL2, accuracyL2 = gradientDescent(X_train, y_train, n, alpha, nIterations, l2=L2)
        print("-" * 20)
        print("L2, Ridge Param: " + str(L2))
        print("-" * 20)

        print("Running L1, Lasso Regression...")
        params = {'alpha': np.linspace(0.5, 1, num=100)}
        rdg_reg = Lasso()
        clf = GridSearchCV(rdg_reg, params, cv=5, scoring = 'neg_mean_squared_error')
        clf.fit(X_train, y_train)
        L1 = round(clf.best_params_['alpha'], 5)
        thetaL1, rmseL1, accuracyL1 = gradientDescent(X_train, y_train, n, alpha, nIterations, l1=L1)
        print("-" * 20)
        print("L1, Lasso Param: " + str(L1))
        print("-" * 20)
        pickle.dump(theta, open('./q2amodels/theta.sav', 'wb'))
        pickle.dump(rmse, open('./q2amodels/rmse.sav', 'wb'))
        pickle.dump(accuracy, open('./q2amodels/accuracy.sav', 'wb'))

        pickle.dump(L2, open('./q2amodels/L2.sav', 'wb'))    
        pickle.dump(thetaL2, open('./q2amodels/thetaL2.sav', 'wb'))    
        pickle.dump(rmseL2, open('./q2amodels/rmseL2.sav', 'wb'))
        pickle.dump(accuracyL2, open('./q2amodels/accuracyL2.sav', 'wb'))

        pickle.dump(L1, open('./q2amodels/L1.sav', 'wb'))    
        pickle.dump(thetaL1, open('./q2amodels/thetaL1.sav', 'wb'))    
        pickle.dump(rmseL1, open('./q2amodels/rmseL1.sav', 'wb'))
        pickle.dump(accuracyL1, open('./q2amodels/accuracyL1.sav', 'wb'))
    else:
        print("Loading Saved Models...")
        theta = pickle.load(open('./q2amodels/theta.sav', 'rb'))
        rmse = pickle.load(open('./q2amodels/rmse.sav', 'rb'))
        accuracy = pickle.load(open('./q2amodels/accuracy.sav', 'rb'))

        L2 = pickle.load(open('./q2amodels/L2.sav', 'rb'))    
        thetaL2 = pickle.load(open('./q2amodels/thetaL2.sav', 'rb'))    
        rmseL2 = pickle.load(open('./q2amodels/rmseL2.sav', 'rb'))
        accuracyL2 = pickle.load(open('./q2amodels/accuracyL2.sav', 'rb'))

        L1 = pickle.load(open('./q2amodels/L1.sav', 'rb'))    
        thetaL1 = pickle.load(open('./q2amodels/thetaL1.sav', 'rb'))    
        rmseL1 = pickle.load(open('./q2amodels/rmseL1.sav', 'rb'))
        accuracyL1 = pickle.load(open('./q2amodels/accuracyL1.sav', 'rb'))

        print("-" * 20)
        print("L2, Ridge Param: " + str(L2))
        print("-" * 20)

        print("-" * 20)
        print("L1, Lasso Param: " + str(L1))
        print("-" * 20)

    print("Computing Accuracies...")
    trainAccuracy, trainAccuracyL1, trainAccuracyL2 = getAccuracy(X_train, y_train, theta, thetaL1, thetaL2)
    valAccuracy, valAccuracyL1, valAccuracyL2 = getAccuracy(X_val, y_val, theta, thetaL1, thetaL2)
    testAccuracy, testAccuracyL1, testAccuracyL2 = getAccuracy(X_test, y_test, theta, thetaL1, thetaL2)

    print("-" * 10 + "" + "-" * 10)    
    print("Comparision of Accuracy in %")
    print(" " * 14,"Logistic Regression".center(20), "L1 Regularisation".center(20), "L2 Regularisation".center(20))
    print("Train".center(14), trainAccuracy.center(20), trainAccuracyL1.center(20), trainAccuracyL2.center(20))
    print("Validation".center(14), valAccuracy.center(20), valAccuracyL1.center(20), valAccuracyL2.center(20))
    print("Test".center(14), testAccuracy.center(20), testAccuracyL1.center(20), testAccuracyL2.center(20))
    print("-" * 20)

    print("Plotting Error Curve...")
    xIteraitons = [x for x in range(1, int(nIterations + 1))]
    plt.plot(xIteraitons, list(rmseL2), 'c')
    plt.plot(xIteraitons, list(rmseL1), 'r')
    plt.xlabel('No. of iterations')
    plt.ylabel('RMSE Values')
    plt.gca().legend(("L2 Error", "L1 Error"))
    plt.title("L1, L2 Regularisation Error vs Iterations")
    fig = plt.gcf()
    fig.canvas.set_window_title('L1, L2 Regularisation Error vs Iterations')
    plt.show()
    print("-" * 20)
    print("Plotting Accuracy Curve...")
    plt.plot(xIteraitons, list(accuracyL2), 'c')
    plt.plot(xIteraitons, list(accuracyL1), 'r')
    plt.xlabel('No. of iterations')
    plt.ylabel('Accuracy Values')
    plt.gca().legend(("L2 Accuracy", "L1 Accuracy"))
    plt.title("L1, L2 Accuracy vs Iterations")
    fig = plt.gcf()
    fig.canvas.set_window_title('L1, L2 Accuracy vs Iterations')
    plt.show()