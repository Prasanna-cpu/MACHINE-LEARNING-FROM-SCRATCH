from sklearn.model_selection import train_test_split
from naivebayes import NaiveBayes
from sklearn.datasets import make_classification
import numpy as np

def accuracy(y_true,y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


X,y=make_classification(n_samples=1000,n_classes=2,n_features=10,random_state=123)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )


nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("Naive Bayes classification accuracy", accuracy(y_test, predictions))