import math

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def calculate_loss(model, X, y):
    # print(model, X, y)
    return 0


def predict(model, X):
    return np.array([0 for i in range(len(X))])


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    input_size = len(X[0])
    w1 = np.random.rand(nn_hdim, input_size)
    b1 = np.random.rand(nn_hdim)
    w2 = np.random.rand(input_size, nn_hdim)
    b2 = np.random.rand(nn_hdim)

    model = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}
    # #print(model)
    A = np.array([0.0 for i in range(nn_hdim)])

    H = np.array([0.0 for i in range(nn_hdim)])
    z = np.zeros_like(X)
    y_pred = np.zeros_like(X)
    # onehot encode
    onehot_encoder = OneHotEncoder(sparse=False)
    y_vector = onehot_encoder.fit_transform(y.reshape(len(y), 1))
    loss = 0

    for x in range(len(X)):
        for i in range(nn_hdim):
            for k in range(len(model['W1'][0])):
                A[i] = A[i] + np.multiply(model['W1'][i][k], X[x][k])
            A[i] = A[i] + b1[i]
            H[i] = math.tanh(A[i])
        for l in range(len(model['W2'])):
            for m in range(len(model['W2'][0])):
                z[x][l] = np.add(z[x][l], np.multiply(model['W2'][l][m], H[l]))
            y_pred[x][l] = math.e ** (z[x][l] + b2[l])
        sum = np.sum(y_pred[x])

        for l in range(len(model['W2'])):
            y_pred[x][l] = y_pred[x][l] / sum

        for l in range(len(model['W2'])):
            loss = loss + (y_vector[x][l] * math.log(y_pred[x][l]))
        # print(y_pred[x],y[x],y_vector[x],loss)
        model = updateModel(model, A, H, X[x], y_pred[x], y_vector[x])
    loss = (-1) * loss / input_size
    # print(loss)

    return model


def updateModel(model, A, H, x, y_pred, y_vector):
    # print("update")
    vdL_dy = dL_dy(y_pred, y_vector)

    vdL_da = dL_da(A, model['W2'], vdL_dy)
    vdL_dw2 = dL_dw2(H, vdL_dy)
    vdL_dw1 = dL_dw1(x, vdL_da)
    vdL_db2 = vdL_dy
    vdL_db1 = vdL_da
    eta=0.1
    print(model["W1"],vdL_dw1)
    w1 = model["W1"] - np.multiply(eta, vdL_dw1)
    w2 = model["W2"] - np.multiply(eta, vdL_dw2)
    b1 = model["b1"] - np.multiply(eta, vdL_db1)
    b2 = model["b2"] - np.multiply(eta, vdL_db2)
    model = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}
    return model


def dL_dy(y_pred, y_vector):
    # #print(y_pred,y_vector)
    # print("dL_dy")
    # print(y_pred,y_vector)
    return np.subtract(y_pred, y_vector)[np.newaxis]


def dL_da(a, w2, vdl_dy):
    # print("dL_da")
    # print(a)
    return np.dot((1 - np.square(np.tanh(a)))[np.newaxis], np.multiply(vdl_dy, w2.transpose()))


def dL_dw2(h, dL_dy):
    print("dL_dw2")
    print(h,dL_dy)
    return np.multiply(np.transpose(h[np.newaxis]), dL_dy)


def dL_db2():
    print("dL_db2")


def dL_dw1(x, dL_da):
    print("dL_dw1")
    # print(x)
    a = x[np.newaxis]
    # print(a)
    # print(a.T)

    return np.multiply(np.transpose(x[np.newaxis]), dL_da)


def dL_db1():
    print("dL_db1")
