#导入svm和数据集
from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
#载入鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

attributes = X
target = y


X_train,X_test,y_train,y_test =  train_test_split( attributes , target , test_size=0.3)
#调用SVC()
clf = svm.SVC()


#fit()训练
clf.fit(X_train,y_train)
#predict()预测
pre_y = clf.predict(X[5:10])
print(pre_y)
print(y[5:10])
#导入numpy
import numpy as np
test = np.array([[5.1,2.9,1.8,3.6]])
#对test进行预测
test_y = clf.predict(test)
print(test_y)
score = clf.score(X_test,y_test)
print(score)