import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


csv = pd.read_csv("~/Documents/ml/ccpp.csv")
Xn = np.array(csv.iloc[:,:4])
one = np.ones(Xn.shape[0])
X = np.column_stack((Xn,one))
y = np.array(csv.iloc[:,4])

m=len(X)

loop_max = 10

epsilon = 0.0001

np.random.seed(1)

theta = np.random.randn(5)

alpha = 0.001

theta_last = np.zeros(5)

count = 0
finish = 0

while count<loop_max:
    count += 1

    sum_m = np.zeros(5)

    for i in range(m):
        gradient = (np.dot(theta,X[i])-y[i])*X[i]
        sum_m = sum_m + gradient

    theta = theta - alpha*sum_m

    if np.linalg.norm(theta-theta*sum_m)<epsilon:
        finish = 1
        break
    else:
        theta_last = theta

    print("loop count = %d" %count  )
    print("theta = %d",theta)





#print(X)