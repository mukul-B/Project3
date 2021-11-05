import math

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.extmath import softmax


def fun(e):
    return math.tanh(e)


tanhVec = np.vectorize(fun)


def fun2(x):
    return math.e ** x


expoVec = np.vectorize(fun2)


def calculate_loss(model, X, y):
    # #print(model, X, y)
    return 0


def predict(model, X):
    y_pred = np.zeros_like(X)
    for x in range(len(X)):
        A = np.matmul(X[x], model['W1']) + model['b1']
        H = tanhVec(A)
        z = np.matmul(H, model['W2']) + model['b2']
        y_pred[x] = z
    y_res = np.array([1 if (y_pred[i][1] > y_pred[i][0]) else 0 for i in range(len(X))])
    return y_res


def random_model(input_size, nn_hdim):
    # np.random.seed(1)
    w1 = np.random.normal(size=(input_size, nn_hdim))
    b1 = np.zeros(nn_hdim)
    w2 = np.random.normal(size=(nn_hdim, input_size))
    b2 = np.random.normal(size=input_size)
    model = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}
    return model


def build_model(X, y, nn_hdim, num_passes=1, print_loss=False):
    input_size = len(X[0])
    output_size = len(X[0])
    model = random_model(input_size, nn_hdim)
    onehot_encoder = OneHotEncoder(sparse=False)
    y_vector = onehot_encoder.fit_transform(y.reshape(len(y), 1))

    for passes in range(num_passes):
        loss = 0
        y_pred = np.zeros_like(X)

        for x in range(len(X)):
            A = np.matmul(X[x], model['W1']) + model['b1']
            H = tanhVec(A)
            z = np.matmul(H, model['W2']) + model['b2']
            y_pred[x] = np.squeeze(softmax(z[np.newaxis]))

            for l in range(output_size):
                if y_pred[x][l] != 0.0:
                    loss = (y_vector[x][l] * math.log(y_pred[x][l]))
            # print(loss)
            # print(model)
            model = updateModel(model, A, H, X[x], y_pred[x], y_vector[x])

        if print_loss:
            loss = (-1) * loss / input_size
            print(loss)

    return model


def updateModel(model, A, H, x, y_pred, y_vector):
    vdL_dy = dL_dy(y_pred, y_vector)
    vdL_da = dL_da(A, model['W2'], vdL_dy)
    vdL_dw2 = dL_dw2(H, vdL_dy)
    vdL_dw1 = dL_dw1(x, vdL_da)
    vdL_db2 = vdL_dy
    vdL_db1 = vdL_da
    eta = 0.05
    w1 = model["W1"] - np.multiply(eta, vdL_dw1)
    w2 = model["W2"] - np.multiply(eta, vdL_dw2)
    b1 = model["b1"] - np.multiply(eta, vdL_db1)
    b2 = model["b2"] - np.multiply(eta, vdL_db2)
    model = {'W1': w1, 'b1': np.squeeze(b1), 'W2': w2, 'b2': np.squeeze(b2)}
    return model


def dL_dy(y_pred, y_vector):
    return np.subtract(y_pred, y_vector)[np.newaxis]


def dL_da(a, w2, vdl_dy):
    return np.multiply((1 - np.square(np.tanh(a)))[np.newaxis], np.squeeze(np.matmul(vdl_dy, w2.transpose())))


def dL_dw2(h, dL_dy):
    return np.squeeze(np.matmul(np.transpose(h[np.newaxis]), dL_dy))


def dL_dw1(x, dL_da):
    return np.matmul(np.transpose(x[np.newaxis]), dL_da)
