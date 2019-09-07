import numpy as np
import matplotlib.pyplot as plt


def gradient(x, y):
    gradX = 2 * (13 * x + 12 * y) / 25;
    gradY = 2 * (12 * x + 13 * y) / 25;
    grad = [gradX, gradY]
    return grad

def minimize(x0, y0, tol, stepSize):
    grad0 = gradient(x0, y0)
    gradX = grad0[0]
    gradY = grad0[1]
    xL = [x0]
    yL = [y0]
    x = x0
    y = y0
    while((abs(gradX) > tol or abs(gradY) > tol)):
        grad = gradient(x, y)
        gradX = grad[0]
        gradY = grad[1]
        x = x - stepSize * gradX
        y = y - stepSize * gradY
        xL.append(x)
        yL.append(y)
        print(grad)
    return xL, yL


def main():
    # Make a simple mesh grid
    x = np.linspace(-100, 100, num = 100)
    y = np.linspace(-100, 100, num = 100)
    xv, yv = np.meshgrid(x, y)

    # Rotate the points in the transform by -45 degrees. 
    xt = xv / np.sqrt(2) + -yv / np.sqrt(2)
    yt = xv / np.sqrt(2) + yv / np.sqrt(2)
    Z = (xt)**2 / 25 + (yt)**2;

    # Observe that the elipse is rotated by 45 degrees. 
    plt.contour(xv, yv, Z)

    # Perform graident descent with an initial value
    x0 = -100
    y0 = 40
    xL, yL = minimize(x0, y0, 0.1, 0.9)
    plt.plot(xL, yL, 'r-o')
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.show()

if __name__=="__main__":
	main()
