from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

data = [[0,0],
       [0,1],
       [1,0],
       [1,1]]
labels_xand = [0, 0, 0, 1]

plt.scatter((data[0]), (data[1]), c = labels_xand)
plt.show()

classifier = Perceptron(max_iter = 40)
classifier.fit(data, labels_xand)
classifier.score(data, labels_xand)

x_values = [np.linspace(0, 1, 100)]
y_values = [np.linspace(0, 1, 100)]


point_grid = list(product(x_values, y_values))

distances = classifier.decision_function(point_grid)
