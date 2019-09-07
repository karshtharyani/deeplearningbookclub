import numpy as np
import matplotlib.pyplot as plt
theta = np.random.randn(1000, 1) * 2 * np.pi;
np.reshape(theta, 1000, 1)
r1 = np.random.rand(1000, 1) * 10 + 5
r2 = np.random.rand(1000, 1) * 25 + 20
x1 = r1 * (np.cos(theta))
y1 = r1 * (np.sin(theta))
x2 = r2 * (np.cos(theta))
y2 = r2 * (np.sin(theta))
plt.scatter(x1, y1)
plt.scatter(x2, y2, c = [[1, 0, 0]])
plt.axis('equal')
plt.show()
plt.clf()
plt.scatter(r1, theta)
plt.scatter(r2, theta, c = [[1, 0, 0]])
plt.show()
