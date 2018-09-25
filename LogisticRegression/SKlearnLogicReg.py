from getData import getdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

attributes,target = getdata()

X_train,X_test,y_train,y_test =  train_test_split( attributes , target , test_size=0.3)

logreg = LogisticRegression(solver="newton-cg",multi_class="multinomial")
logreg.fit(X_train,y_train)

acc = logreg.score(X_test,y_test)
print(acc)


