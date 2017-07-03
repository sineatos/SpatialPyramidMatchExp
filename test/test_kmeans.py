# -*- encoding:UTF-8 -*-
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
import numpy as np
import matplotlib.pylab as plt

points = scipy.randn(20, 4)
print(points)
data = whiten(points)
# print(data)

centroid = kmeans(data, 3)[0]
print(centroid)

label = vq(np.array([[1, 1, 1, 1]]), centroid)

print(label)
