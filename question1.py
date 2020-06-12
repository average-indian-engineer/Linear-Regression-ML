import numpy as np
import matplotlib.pyplot as plt 
from random import randrange
import copy 
import csv
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
import warnings
import pickle

# Column Transformations
mapSex = lambda x : 0 if x == "I" else (1 if x == "M" else -1)
scaleElement = lambda x, min, max : (x-min)/(max-min)

# Function to Apply column transformations
def applyTransformation(arr, column, transformation):
    arr[:,column] = [transformation(x) for x in arr[:,column]]

# Read data from file
def readData(filename):
    file = open(filename, "r")
    rawData = []
    for line in file:
        rawData.append(line.split())
    return np.array(rawData)

# Normalize data 
def minMaxScale(X, m, n):
    for j in range(n-1):
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

# Generate KFold splits 
def generateKFolds(dataset, folds):
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

# Hypothesis
def hypothesis(theta, X):
    return np.matmul(X, theta)

def getRmseCost(predict, actual):
    m = predict.shape[0]
    return (sum(np.square(np.subtract(predict, actual))) / m) ** 0.5

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

def gradientDescent(X, y, X_test, y_test, n, alpha, nIterations, l1=None, l2=None):
    cost = np.zeros(nIterations)
    cost_test = np.zeros(nIterations)
    m = X.shape[0]
    m_test = X_test.shape[0]
    theta = np.zeros((n, 1))
    h = hypothesis(theta, X)
    h_test = hypothesis(theta, X_test)
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
        h_test = hypothesis(theta, X_test)
        cost[i] = getRmseCost(h, y)
        cost_test[i] = getRmseCost(h_test, y_test)
    return theta, cost, cost_test

# Train and Test cost applying linear regression
def linearRegression(X_train, y_train, X_test, y_test, n, alpha, nIterations):
    return gradientDescent(X_train, y_train, X_test, y_test, n, alpha, nIterations)

def normalEquation(X, y):
    X_t = np.transpose(X)
    A = np.matmul(X_t, X)
    B = np.matmul(X_t, y)
    return np.matmul(np.linalg.inv(A), B)

def linearRegressionLoader(useSavedModels):
    print("-" * 10 + "Part A" + "-" * 10)
    print("Collecting data...")

    rawData = readData("./q1dataset/abalone.data")
    m, n = rawData.shape
    # Clean Data
    applyTransformation(rawData, 0, mapSex)
    rawData = rawData.astype('float64')
    normalData = minMaxScale(rawData, m, n-1)
    # Create KFolds
    nIterations = 500
    alpha = 0.1
    kFold = 5 
    print("Generating splits...")
    splitData = generateKFolds(normalData, kFold)
    listSplits = list(splitData)

    # Apply Linear Regression on KFolds
    
    rmseLinear_train = []
    rmseLinear_test = []
    
    rmseNormal_train = []
    rmseNormal_test = []
    normalThetas = []

    if(not useSavedModels):
        print("Running Model...")
        for i in range(kFold):
            testData = np.array(listSplits[i])
            trainSplits = listSplits[:i] + listSplits[i+1:]
            trainData = []
            for split in trainSplits:
                for element in split:
                    trainData.append(element)
            trainData = np.array(trainData)

            # Split output & input
            X_train, y_train = trainData[:,:n-1], trainData[:,n-1]
            X_test, y_test = testData[:,:n-1], testData[:,n-1]
            
            # Append 1s in start
            oneCol = np.ones((X_train.shape[0], 1))
            X_train = np.concatenate((oneCol, X_train), axis=1)
            oneCol = np.ones((X_test.shape[0], 1))
            X_test = np.concatenate((oneCol, X_test), axis=1)

            y_train = y_train.reshape((X_train.shape[0], 1))
            y_test = y_test.reshape((X_test.shape[0], 1))

            theta, rmse_train, rmse_test = linearRegression(X_train, y_train, X_test, y_test, n, alpha, nIterations)
            rmseLinear_train.append(rmse_train)
            rmseLinear_test.append(rmse_test)

            thetaNormal = normalEquation(X_train, y_train)
            normalThetas.append(thetaNormal)
            hNormal_train = hypothesis(thetaNormal, X_train)
            hNormal_test = hypothesis(thetaNormal, X_test)
            rmseNormal_train.append(getRmseCost(hNormal_train, y_train))
            rmseNormal_test.append(getRmseCost(hNormal_test, y_test))
        pickle.dump(rmseLinear_train, open('./q1amodels/rmseLinear_train.sav', 'wb'))
        pickle.dump(rmseLinear_test, open('./q1amodels/rmseLinear_test.sav', 'wb'))
        pickle.dump(rmseNormal_train, open('./q1amodels/rmseNormal_train.sav', 'wb'))
        pickle.dump(rmseNormal_test, open('./q1amodels/rmseNormal_test.sav', 'wb'))
        pickle.dump(normalThetas, open('./q1amodels/normalThetas.sav', 'wb'))
    else:
        print("Loading Saved Models...")
        rmseLinear_train = pickle.load(open('./q1amodels/rmseLinear_train.sav', 'rb'))
        rmseLinear_test = pickle.load(open('./q1amodels/rmseLinear_test.sav', 'rb'))

        rmseNormal_train = pickle.load(open('./q1amodels/rmseNormal_train.sav', 'rb'))
        rmseNormal_test = pickle.load(open('./q1amodels/rmseNormal_test.sav', 'rb'))
        normalThetas = pickle.load(open('./q1amodels/normalThetas.sav', 'rb'))

    meanRmse_train = np.mean(rmseLinear_train, axis = 0)
    meanRmse_test = np.mean(rmseLinear_test, axis = 0)

    print("Plotting curves...")
    xIteraitons = [x for x in range(1, int(nIterations + 1))]
    plt.plot(xIteraitons, list(meanRmse_train), 'b')
    plt.plot(xIteraitons, list(meanRmse_test), 'r')
    plt.xlabel('No. of iterations')
    plt.ylabel('Mean RMSE Values')
    plt.gca().legend(("Training Set", "Testing Set"))
    plt.title("Gradient Descent Error vs Iterations")
    fig = plt.gcf()
    fig.canvas.set_window_title('Gradient Descent Error vs Iterations')
    plt.show()
    print("Press Enter", end="")
    input()
    print("-" * 10 + "Part B" + "-" * 10)
    print("Normal Equation RMSE")
    for i in range(len(normalThetas)):
        print("Train: " + str(rmseNormal_train[i]) + " Test: " + str(rmseNormal_test[i]))
    
    print("Press Enter", end="")
    input()

    print("-" * 10 + "Part C" + "-" * 10)
    print("Comparision of both RMSE Values")
    print(" " * 10,"Gradient Descent".center(25), "Normal Equation".center(20))
    print("Train".center(10), str(round(meanRmse_train[-1], 7)).center(25), str(round(np.mean(rmseNormal_train), 7)).center(20))
    print("Test".center(10), str(round(meanRmse_test[-1], 7)).center(25), str(round(np.mean(rmseNormal_test), 7)).center(20))
    
def regularisationLoader(useSavedModels):
    
    print("-" * 10 + "" + "-" * 10)
    print("Collecting data...")
    rawData = readData("./q1dataset/abalone.data")
    m, n = rawData.shape
    # Clean Data
    applyTransformation(rawData, 0, mapSex)
    rawData = rawData.astype('float64')
    normalData = minMaxScale(rawData, m, n-1)
    # Create KFolds
    nIterations = 500
    alpha = 0.1
    kFold = 5 
    print("Generating splits...")
    splitData = generateKFolds(normalData, kFold)
    listSplits = list(splitData)
    # Apply Linear Regression on KFolds
    rmseLinear_train = []
    rmseLinear_test = []
    rmseNormal_train = []
    rmseNormal_test = []
    normalThetas = []

    if(not useSavedModels):
        print("Running Model...")
        for i in range(kFold):
            testData = np.array(listSplits[i])
            trainSplits = listSplits[:i] + listSplits[i+1:]
            trainData = []
            for split in trainSplits:
                for element in split:
                    trainData.append(element)
            trainData = np.array(trainData)

            # Split output & input
            X_train, y_train = trainData[:,:n-1], trainData[:,n-1]
            X_test, y_test = testData[:,:n-1], testData[:,n-1]
            
            # Append 1s in start
            oneCol = np.ones((X_train.shape[0], 1))
            X_train = np.concatenate((oneCol, X_train), axis=1)
            oneCol = np.ones((X_test.shape[0], 1))
            X_test = np.concatenate((oneCol, X_test), axis=1)

            y_train = y_train.reshape((X_train.shape[0], 1))
            y_test = y_test.reshape((X_test.shape[0], 1))

            theta, rmse_train, rmse_test = linearRegression(X_train, y_train, X_test, y_test, n, alpha, nIterations)
            rmseLinear_train.append(rmse_train)
            rmseLinear_test.append(rmse_test)

            thetaNormal = normalEquation(X_train, y_train)
            normalThetas.append(thetaNormal)
            hNormal_train = hypothesis(thetaNormal, X_train)
            hNormal_test = hypothesis(thetaNormal, X_test)
            rmseNormal_train.append(getRmseCost(hNormal_train, y_train))
            rmseNormal_test.append(getRmseCost(hNormal_test, y_test))
        pickle.dump(rmseLinear_train, open('./q1bmodels/rmseLinear_train.sav', 'wb'))
        pickle.dump(rmseLinear_test, open('./q1bmodels/rmseLinear_test.sav', 'wb'))
        pickle.dump(rmseNormal_train, open('./q1bmodels/rmseNormal_train.sav', 'wb'))
        pickle.dump(rmseNormal_test, open('./q1bmodels/rmseNormal_test.sav', 'wb'))
        pickle.dump(normalThetas, open('./q1bmodels/normalThetas.sav', 'wb'))
    else:
        print("Loading Saved Models...")
        rmseLinear_train = pickle.load(open('./q1bmodels/rmseLinear_train.sav', 'rb'))
        rmseLinear_test = pickle.load(open('./q1bmodels/rmseLinear_test.sav', 'rb'))
        rmseNormal_train = pickle.load(open('./q1bmodels/rmseNormal_train.sav', 'rb'))
        rmseNormal_test = pickle.load(open('./q1bmodels/rmseNormal_test.sav', 'rb'))
        normalThetas = pickle.load(open('./q1bmodels/normalThetas.sav', 'rb'))

    print("Getting RMSE...")
    xIteraitons = [x for x in range(1, int(nIterations + 1))]
    rmseLinear_train = np.array(rmseLinear_train)
    minSplitIndex = np.argmin(rmseLinear_train[:,-1])
    
    print("Splitting 80%...")
    splittedDataset = listSplits[:minSplitIndex] + listSplits[minSplitIndex+1:] 
    sparceDataset = []
    for split in splittedDataset:
        for element in split:
            sparceDataset.append(element)
    
    sparceDataset = np.array(sparceDataset)
    X, y = sparceDataset[:,:n-1], sparceDataset[:,n-1]
    oneCol = np.ones((X.shape[0], 1))
    X = np.concatenate((oneCol, X), axis=1)
    y = y.reshape((X.shape[0], 1))

    L2 = None
    rmseL2 = None
    rmseL2_test = None
    if(not useSavedModels):
        print("Running L2, Ridge...")
        print("-" * 10 + "Part A" + "-" * 10)
        params = {'alpha': np.linspace(0.1, 1.0, num=200)}
        rdg_reg = Ridge()
        clf = GridSearchCV(rdg_reg, params, cv=5, scoring = 'neg_mean_squared_error')
        clf.fit(X, y)
        L2 = round(clf.best_params_['alpha'], 5)
        print("L2, Ridge Param: " + str(L2))
    else:
        L2 = pickle.load(open('./q1bmodels/L2.sav', 'rb'))
        rmseL2 = pickle.load(open('./q1bmodels/rmseL2.sav', 'rb'))
        rmseL2_test = pickle.load(open('./q1bmodels/rmseL2_test.sav', 'rb'))
        print("-" * 10 + "Part A" + "-" * 10)
        print("L2, Ridge Param: " + str(L2))
    
    L1 = None
    rmseL1 = None
    rmseL1_test = None
    if(not useSavedModels):
        print("Running L1, Lasso...")
        print("-" * 10 + "Part B" + "-" * 10)
        params = {'alpha': np.linspace(0.0001, 0.005, num=100)}
        rdg_reg = Lasso()
        clf = GridSearchCV(rdg_reg, params, cv=5, scoring = 'neg_mean_squared_error')
        clf.fit(X, y)
        L1 = round(clf.best_params_['alpha'], 5)
        print("L1, Lasso Param: " + str(L1))
    else:
        L1 = pickle.load(open('./q1bmodels/L1.sav', 'rb'))
        rmseL1 = pickle.load(open('./q1bmodels/rmseL1.sav', 'rb'))
        rmseL1_test = pickle.load(open('./q1bmodels/rmseL1_test.sav', 'rb'))
        print("-" * 10 + "Part B" + "-" * 10)
        print("L1, Lasso Param: " + str(L2))
    
    print("-" * 10 + "" + "-" * 10)    
    testDataset = listSplits[minSplitIndex]
    X_test, y_test = testDataset[:,:n-1], testDataset[:,n-1]
    oneCol = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((oneCol, X_test), axis=1)
    y_test = y_test.reshape((X_test.shape[0], 1))

    if(not useSavedModels):
        print("Getting L1 Theta...")
        theta, rmseL1, rmseL1_test = gradientDescent(X, y, X_test, y_test, n, alpha, nIterations, l1=L1)
        pickle.dump(L1, open('./q1bmodels/L1.sav', 'wb'))
        pickle.dump(rmseL1, open('./q1bmodels/rmseL1.sav', 'wb'))
        pickle.dump(rmseL1_test, open('./q1bmodels/rmseL1_test.sav', 'wb'))

    print("Test RMSE with L1: " + str(round(rmseL1_test[-1], 7)))
    print("Plotting L1 Curve...")

    plt.plot(xIteraitons, list(rmseL1), 'b')
    plt.plot(xIteraitons, list(rmseL1_test), 'r')
    plt.xlabel('No. of iterations')
    plt.ylabel('L1 Regularisation RMSE Values')
    plt.gca().legend(("Training Set", "Testing Set"))
    plt.title("L1 Regularisation Error vs Iterations")
    fig = plt.gcf()
    fig.canvas.set_window_title('L1 Regularisation Error vs Iterations')
    plt.show()
    print("-" * 10 + "" + "-" * 10)

    if(not useSavedModels):
        print("Getting L2 Theta...")
        theta, rmseL2, rmseL2_test = gradientDescent(X, y, X_test, y_test, n, alpha, nIterations, l2=L2)
        pickle.dump(L2, open('./q1bmodels/L2.sav', 'wb'))
        pickle.dump(rmseL2, open('./q1bmodels/rmseL2.sav', 'wb'))
        pickle.dump(rmseL2_test, open('./q1bmodels/rmseL2_test.sav', 'wb'))

    print("Test RMSE with L2: " + str(round(rmseL2_test[-1], 7)))

    print("Plotting L2 Curve...")
    plt.plot(xIteraitons, list(rmseL2), 'b')
    plt.plot(xIteraitons, list(rmseL2_test), 'r')
    plt.xlabel('No. of iterations')
    plt.ylabel('L2 Regularisation RMSE Values')
    plt.gca().legend(("Training Set", "Testing Set"))
    plt.title("L2 Regularisation Error vs Iterations")
    fig = plt.gcf()
    fig.canvas.set_window_title('L2 Regularisation Error vs Iterations')
    plt.show()

def readCSV(filename):
    raw = []
    with open(filename, 'rt') as f:
        data = csv.reader(f)
        for row in data:
            raw.append(row)
    return raw

def linearRegressionLine(X, Y, n, alpha, nIterations):
    theta = np.zeros((n, 1))
    h = hypothesis(theta, X)
    return gradientDescentLine(theta, h, X, Y, n, alpha, nIterations)


def gradientDescentRegularizedLine(X, y, n, alpha, nIterations, l1=None, l2=None):
    m = X.shape[0]
    theta = np.zeros((n, 1))
    h = hypothesis(theta, X)
    for i in range(nIterations):
        oldTheta = copy.deepcopy(theta)
        oldTheta[0] = 0
        if(l1):
            theta = theta - (alpha/m) * np.transpose(np.matmul(np.transpose(h - y), X)) - (l1 * alpha / m) / 2 * signBits(oldTheta)
        else:
            theta = theta - (alpha/m) * np.transpose(np.matmul(np.transpose(h - y), X)) - (l2 * alpha / m) * oldTheta
        h = hypothesis(theta, X)
    return theta
    
def gradientDescentLine(theta, h, X, y, n, alpha, nIterations):
    m = X.shape[0]
    for i in range(nIterations):
        theta = theta - ( (alpha/m) * np.matmul(np.transpose(X), h - y))
        h = hypothesis(theta, X)
    return theta

def bestFitLineLoader(useSavedModels):
    warnings.simplefilter("ignore")
    print("-" * 10 + "Part A" + "-" * 10)
    print("Collecting data...")
    rawData = readCSV('./q1dataset/brainbody.csv')[1:]
    rawData = np.array(rawData)
    rawData = rawData.astype('float64')
    X, Y = rawData[:,0], rawData[:,1]

    print("Cleaning Data...")
    X = X.reshape((X.shape[0], 1))
    Y = Y.reshape((Y.shape[0], 1))

    oneCol = np.ones((X.shape[0], 1))
    X = np.concatenate((oneCol, X), axis=1)
    m, n = X.shape
    alpha = 0.00001
    nIterations = 300
    
    theta = None
    if(not useSavedModels):
        print("Running Gradient Model...")
        theta = linearRegressionLine(X, Y, n, alpha, nIterations)
        pickle.dump(theta, open('./q1cmodels/theta.sav', 'wb'))
    else:
        print("Loading Saved Gradient Model...")
        theta = pickle.load(open('./q1cmodels/theta.sav', 'rb'))

    print("Plotting Gradient Descent Curve...")
    x = np.linspace(0, 100, 500)
    y = theta[0] + theta[1] * x
    plt.plot(x, y, 'b')
    plt.scatter(X[:,1], Y, 0.6, 'r')
    
    plt.xlabel('Brain Weight')
    plt.ylabel('Body Weight')
    plt.gca().legend(("Gradient Descent", "Scatter Plot"))
    plt.title("Body Weight vs Brain Weight with Gradient Descent")
    fig = plt.gcf()
    fig.canvas.set_window_title('Body Weight vs Brain Weight with Gradient Descent')
    plt.show()

    print("-" * 10 + "Part B" + "-" * 10)
    L2 = None
    thetaL2 = None
    if(not useSavedModels):
        print("Running L2 Regularisation...")
        params = {'alpha': np.linspace(0.00008, 0.000015, num=100)}
        rdg_reg = Ridge()
        clf = GridSearchCV(rdg_reg, params, cv=5, scoring = 'neg_mean_squared_error')
        clf.fit(X, Y)
        L2 = round(clf.best_params_['alpha'], 5)
        pickle.dump(L2, open('./q1cmodels/L2.sav', 'wb'))
    else:
        L2 = pickle.load(open('./q1cmodels/L2.sav', 'rb'))
    print("L2, Ridge Param: " + str(L2))
    if(not useSavedModels):
        thetaL2 = gradientDescentRegularizedLine(X, Y, n, alpha, nIterations, l2=L2)
        pickle.dump(thetaL2, open('./q1cmodels/thetaL2.sav', 'wb'))
    else:
        print("Loading Saved L2 Model...")
        thetaL2 = pickle.load(open('./q1cmodels/thetaL2.sav', 'rb'))

    print("Plotting L2 Curve...")
    x = np.linspace(0, 100, 500)
    y = thetaL2[0] + thetaL2[1] * x
    plt.plot(x, y, 'b')
    plt.scatter(X[:,1], Y, 0.6, 'r')
    
    plt.xlabel('Brain Weight')
    plt.ylabel('Body Weight')
    plt.gca().legend(("L2 Plot", "Scatter Plot"))
    plt.title("Body Weight vs Brain Weight with L2 Regularisation")
    fig = plt.gcf()
    fig.canvas.set_window_title('Body Weight vs Brain Weight with L2 Regularisation')
    plt.show()

    print("-" * 10 + "Part C" + "-" * 10)
    L1 = None
    thetaL1 = None
    if(not useSavedModels):
        print("Running L1 Regularisation...")
        params = {'alpha': np.linspace(0.00009, 0.000015, num=100)}
        rdg_reg = Lasso()
        clf = GridSearchCV(rdg_reg, params, cv=5, scoring = 'neg_mean_squared_error')
        clf.fit(X, Y)
        L1 = round(clf.best_params_['alpha'], 5)
        pickle.dump(L1, open('./q1cmodels/L1.sav', 'wb'))
    else:
        L2 = pickle.load(open('./q1cmodels/L2.sav', 'rb'))
    print("L1, Lasso Param: " + str(L1))
    if(not useSavedModels):
        thetaL1 = gradientDescentRegularizedLine(X, Y, n, alpha, nIterations, l1=L1)
        pickle.dump(thetaL1, open('./q1cmodels/thetaL1.sav', 'wb'))
    else:
        print("Loading Saved L1 Model...")
        thetaL1 = pickle.load(open('./q1cmodels/thetaL1.sav', 'rb'))
    
    print("Plotting L1 Curve...")
    x = np.linspace(0, 100, 500)
    y = thetaL1[0] + thetaL1[1] * x
    plt.plot(x, y, 'b')
    plt.scatter(X[:,1], Y, 0.6, 'r')
    
    plt.xlabel('Brain Weight')
    plt.ylabel('Body Weight')
    plt.gca().legend(("L1 Plot", "Scatter Plot"))
    plt.title("Body Weight vs Brain Weight with L1 Regularisation")
    fig = plt.gcf()
    fig.canvas.set_window_title('Body Weight vs Brain Weight with L1 Regularisation')
    plt.show()
    