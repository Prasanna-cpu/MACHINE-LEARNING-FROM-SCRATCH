import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

from logistic import LogisticRegression


bc=load_breast_cancer()
X,y=bc.data,bc.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


model=LogisticRegression()
model.fit(X_train, y_train)
model.predict(X_test)
y_pred=model.predict(X_test)

def accuracy(y_pred,y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc=accuracy(y_pred,y_test)
print(f"Accuracy:{acc}")
