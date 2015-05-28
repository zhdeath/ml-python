"""
  a K-nearest neighbors (KNN) classification model with the iris flower dataset
"""

# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X"
X = iris.data
# store response vector in "y"
y = iris.target

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# fit the model with data(training model)
knn.fit(X, y)

# predict the response for a new observation
print(knn.predict([3,5,4,2]))
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
print(knn.predict(X_new))
