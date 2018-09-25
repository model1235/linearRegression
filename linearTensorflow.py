import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing


csv = pd.read_csv("~/Documents/ml/ccpp.csv")
X_train = np.array(csv.iloc[:,:4])
#one = np.ones(Xn.shape[0])
#X = np.column_stack((Xn,one))
y_train = np.array(csv.iloc[:,4])

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
rate = 0.00001
y_train=y_train.reshape(-1,1)

X = tf.placeholder("float",[None,4])
y = tf.placeholder("float",[None,1])

w=tf.Variable(tf.zeros([4,1]))
b = tf.Variable(tf.zeros([1]))

print(X_train)
pred = tf.matmul(X,w)+b
loss= tf.reduce_mean(tf.reduce_mean(pred-y))
optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for i in range(10000):
    sess.run(optimizer,{X:X_train,y:y_train})
#sess.run(loss,{pX:X,py:y})
wr = [sess.run(w[0]),sess.run(w[1]),sess.run(w[2]),sess.run(w[3])]
#wr = sess.run(w)
b=sess.run(b)

print("------------------")
print(min_max_scaler.scale_)
print(wr)

print(b)
