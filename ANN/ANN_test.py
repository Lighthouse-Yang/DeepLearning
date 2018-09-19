import numpy as np

# input
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output
y = np.array([[0.1],
              [0.8],
              [0.86],
              [0.23]])
# weight
np.random.seed(1)

w0 = 2*np.random.random((3, 4)) - 1
w1 = 2*np.random.random((4, 1)) - 1


# Nonlinear function
def sigmoid(X, derive=False):
    if not derive:
        return 1 / (1 + np.exp(-X))
    else:
        return X * (1 - X)


# Training
training_time = 6000
for i in range(training_time):
    # Layer0
    A0 = np.dot(X, w0)
    Z0 = sigmoid(A0)

    # Layer1
    A1 = np.dot(Z0, w1)
    # _y实际输出
    _y = Z1 = sigmoid(A1)

    # cost误差
    cost = _y - y     # cost = (y - _y)**2/2
    print('cost: {}'.format(np.mean(np.abs(cost))))

    # calc delta
    delta_A1 = cost * sigmoid(Z1, derive=True)
    delta_w1 = np.dot(Z0.T, delta_A1)
    delta_A0 = np.dot(delta_A1, w1.T) * sigmoid(Z0, derive=True)
    delta_w0 = np.dot(X.T, delta_A0)

    # update
    rate = 1
    w1 = w1 - rate * delta_w1
    w0 = w0 - rate * delta_w0
print(_y)
print(w0, w1)