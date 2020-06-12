import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
import copy
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt 

def setFirstColumn(data, x):
    data = np.array(data)
    m, n = data.shape
    data[:,0] = np.ones(m) * x
    return data

def groupColumn(data, x):
    data = np.copy(data)
    data[:,0] = [e == x for e in data[:,0]]
    return data

def getAccuracy(pred, act):
    arr = []
    m = len(pred)
    for i in range(m):
        if(pred[i] == act[i]):
            arr.append(1)
        else:
            arr.append(0)
    return round((sum(arr) / m)*100, 3)

def saveData():
    for cl in classes:
        filenameL1 = 'modelL1' + str(cl) + '.sav'
        filenameL2 = 'modelL2' + str(cl) + '.sav'
        pickle.dump(modelsL1[cl], open(filenameL1, 'wb'))
        pickle.dump(modelsL2[cl], open(filenameL2, 'wb'))

def mnistLoader(useSavedModels):
    print("-" * 10 + "" + "-" * 10)  
    print("Collecting Data...")
    train_data = pd.read_csv('./q2dataset/mnist_train.csv', header=None)
    test_data = pd.read_csv('./q2dataset/mnist_test.csv', header=None)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    modelsL1 = []
    modelsL2 = []
    sortedData = train_data[train_data[:,0].argsort()]
    classes = np.unique(train_data[:,0])
    sortedData = np.split(sortedData, np.where(np.diff(sortedData[:,0]))[0]+1)
    print("Generating Splits...")
    accuracyL1_train = [0] * len(classes)
    accuracyL1_test = [0] * len(classes)
    accuracyL2_train = [0] * len(classes)
    accuracyL2_test = [0] * len(classes)
    if(not useSavedModels):
        print("Running L1 and L2...")  
        print("-" * 10 + "" + "-" * 10)  
        for cl in classes:
            computeData = copy.deepcopy(sortedData)
            data_1 = np.array(computeData[cl])
            data_0 = np.array(computeData[:cl] + computeData[cl + 1:])
            _data = []
            
            for grp in data_0:
                for row in grp:
                    _data.append(row)
            
            data_1 = setFirstColumn(data_1, 1)
            data_0 = setFirstColumn(_data, 0)

            
            dataset_test = groupColumn(test_data, cl)
            dataset = np.append(data_0, data_1, axis=0)
            X = dataset[:,1:]
            y = dataset[:,0]
            X_test = dataset_test[:,1:]
            y_test = dataset_test[:,0]

            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            X_test = scaler.fit_transform(X_test)

            lRL1 = LogisticRegression(solver='saga', penalty='l1', C=0.1, max_iter=300, tol=0.01)
            lRL1.fit(X, y)
            modelsL1.append(lRL1)

            accuracyL1_train[cl] = round(lRL1.score(X, y) * 100, 3)
            accuracyL1_test[cl] = round(lRL1.score(X_test, y_test) * 100, 3)
            
            lRL2 = LogisticRegression(solver='lbfgs', penalty='l2', C=0.1, max_iter=200, tol=0.01)
            lRL2.fit(X, y)
            modelsL2.append(lRL2)
            accuracyL2_train[cl] = round(lRL2.score(X, y) * 100, 3)
            accuracyL2_test[cl] = round(lRL2.score(X_test, y_test) * 100, 3)

            print("Class " + str(cl) + " Accuracy for L1, L2 in %")
            print(" "*10, "L1 Regularisation".center(20), "L2 Regularisation".center(20))
            print("Train".center(10), str(accuracyL1_train[cl]).center(20), str(accuracyL2_train[cl]).center(20))
            print("Test".center(10), str(accuracyL1_test[cl]).center(20), str(accuracyL2_test[cl]).center(20))
            print("-" * 20)
        for cl in classes:
            filenameL1 = './q2bcmodels/modelL1' + str(cl) + '.sav'
            filenameL2 = './q2bcmodels/modelL2' + str(cl) + '.sav'
            pickle.dump(modelsL1[cl], open(filenameL1, 'wb'))
            pickle.dump(modelsL2[cl], open(filenameL2, 'wb'))
    else:
        print("Loading Saved Models...")  
        print("-" * 10 + "" + "-" * 10)  
        for cl in classes:
            filenameL1 = './q2bcmodels/modelL1' + str(cl) + '.sav'
            filenameL2 = './q2bcmodels/modelL2' + str(cl) + '.sav'
            modelsL1.append(pickle.load(open(filenameL1, 'rb')))
            modelsL2.append(pickle.load(open(filenameL2, 'rb')))    

    scaler = MinMaxScaler()
    X_train = train_data[:,1:]
    y_train = train_data[:,0].ravel()
    X_train = scaler.fit_transform(X_train)

    X_test = test_data[:,1:]
    y_test = test_data[:,0].ravel()
    X_test = scaler.fit_transform(X_test)

    y_train_pred_L1 = []
    y_train_pred_L2 = []
    
    for i in range(X_train.shape[0]):    
        scoreL1_train = [0] * len(classes)
        scoreL2_train = [0] * len(classes)
        for j in classes:
            scoreL1_train[j] = modelsL1[j].predict_proba(X_train[i].reshape(1, -1))[0][1]
            scoreL2_train[j] = modelsL2[j].predict_proba(X_train[i].reshape(1, -1))[0][1]
        y_train_pred_L1.append(np.argmax(scoreL1_train))
        y_train_pred_L2.append(np.argmax(scoreL2_train))
    
    L1_train = getAccuracy(y_train_pred_L1, list(y_train))
    L2_train = getAccuracy(y_train_pred_L2, list(y_train))

    y_test_pred_L1 = []
    y_test_pred_L2 = []
    
    for i in range(X_test.shape[0]):    
        scoreL1_test = [0] * len(classes)
        scoreL2_test = [0] * len(classes)
        for j in classes:
            scoreL1_test[j] = modelsL1[j].predict_proba(X_test[i].reshape(1, -1))[0][1]
            scoreL2_test[j] = modelsL2[j].predict_proba(X_test[i].reshape(1, -1))[0][1]
        y_test_pred_L1.append(np.argmax(scoreL1_test))
        y_test_pred_L2.append(np.argmax(scoreL2_test))
    
    L1_test = getAccuracy(y_test_pred_L1, list(y_test))
    L2_test = getAccuracy(y_test_pred_L2, list(y_test))

    print("Overall accuracy for L1, L2 in %")
    print(" "*10, "L1 Regularisation".center(20), "L2 Regularisation".center(20))
    print("Train".center(10), str(L1_train).center(20), str(L2_train).center(20))
    print("Test".center(10), str(L1_test).center(20), str(L2_test).center(20))
    print("-" * 20)

    n_classes = len(classes)
    scaler = MinMaxScaler()
    xTest = scaler.fit_transform(test_data[:,1:])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        ytest = groupColumn(test_data, i)[:,0]
        yscore = modelsL2[i].predict_proba(xTest)[:,1]
        fpr[i], tpr[i], _ = roc_curve(ytest, yscore)
        roc_auc[i] = auc(fpr[i], tpr[i])

    print("Plotting ROC Curves...")
    # Plot of a ROC curve for a specific class
    plt.figure()    
    for i in classes:
        plt.plot(fpr[i], tpr[i], label='Class ' + str(i) + ' (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for All Classes')
    fig = plt.gcf()
    fig.canvas.set_window_title('ROC Curve for All Classes')
    plt.show()
