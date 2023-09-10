import matplotlib.pyplot as plt
import numpy as np
from PCA import PrincipalComponentAnalysis
from sklearn.datasets import load_iris



data=load_iris()
X=data.data
y=data.target


pca=PrincipalComponentAnalysis(2)
pca.fit(X)
X_proj=pca.transform(X)

print(f"Shape of X:{X.shape}")
print(f"Shape of transformed X:{X_proj.shape}")
