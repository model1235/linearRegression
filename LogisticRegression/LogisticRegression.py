from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from getData import getdata



attributes,target = getdata()




def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


def trainLogRegres(train_x, train_y, maxIter,alpha):
    numSamples, numFeatures = np.shape(train_x)
    weights = np.ones((numFeatures, 1))
    for k in range(maxIter):
        for i in range(numSamples):
            output = sigmoid(train_x[i, :] * weights)
            #print("output")
            #print(output)
            error = train_y[i, 0] - output
            print(weights)
            weights = weights + alpha * train_x[i, :].transpose() * error
    return weights

def testLogRegres(weights, test_x, test_y):
    numSamples, numFeatures = np.shape(test_x)
    matchCount = 0
    for i in range(numSamples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy

X_train,X_test,y_train,y_test =  train_test_split( attributes , target , test_size=0.3)
#print(X_test)
weight = trainLogRegres(X_train,y_train,500,0.001)
print(testLogRegres(weight,X_test,y_test))