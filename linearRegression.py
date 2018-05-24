import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


csv = pd.read_csv("~/Documents/ml/ccpp.csv")
Xn = np.array(csv.iloc[:,:4])
one = np.ones(Xn.shape[0])
X = np.column_stack((Xn,one))
y = np.array(csv.iloc[:,4])
#theta = np.array();
#plt.plot(X[:,0],X[:,1])
#plt.show()

#print()
X2 = X.T.dot(X)
theta = np.linalg.inv(X2).dot(X.T).dot(y)
print(theta)
print("--------------------")
#print(csv.head())
#X = np.array(csv)


regr = LinearRegression()
res = regr.fit(Xn,y)
print(res.coef_)


csv22 = pd.read_csv("~/Documents/ml/ccpp2.csv")
Xn22 = np.array(csv22.iloc[:,:4])
one22 = np.ones(Xn22.shape[0])
X22 = np.column_stack((Xn22,one22))
y22 = np.array(csv22.iloc[:,4])

score = res.score(Xn22,y22)
print(score)

