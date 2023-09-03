from SimpleITK import Mean
from linear import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

X,y=make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.75,random_state=4)

fig=plt.figure(figsize=(10,4))
plt.scatter(X[:,0],y,color='b',marker='o',s=30)
plt.show()

lin=LinearRegression()
lin.fit(X,y)
pred=lin.predict(X_test)

def MeanSquaredError(y_test,pred):
    return np.mean((y_test-pred)**2)

mse=MeanSquaredError(y_test,pred)
print(f"Mean Squared Error:{mse}")