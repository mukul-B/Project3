import math

import numpy as np
from sklearn.preprocessing import OneHotEncoder

def calculate_loss(model, X, y):
    print(model, X, y)
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
    # print(model)
    H = np.array([0.0 for i in range(nn_hdim)])
    z = np.zeros_like(X)
    y_pred = np.zeros_like(X)
    # onehot encode
    onehot_encoder = OneHotEncoder(sparse=False)
    y_vector = onehot_encoder.fit_transform(y.reshape(len(y), 1))
    loss=0
    for x in range(len(X)):
        for i in range(nn_hdim):
            for k in range(len(model['W1'][0])):
                H[i] = H[i] + np.multiply(model['W1'][i][k], X[x][k])
            H[i] = math.tanh(H[i] + b1[i])
        for l in range(len(model['W2'])):
            for m in range(len(model['W2'][0])):
                z[x][l] = np.add(z[x][l], np.multiply(model['W2'][l][m], H[l]))
            y_pred[x][l] = math.e ** (z[x][l] + b2[l])
        sum = np.sum(y_pred[x])

        for l in range(len(model['W2'])):
            y_pred[x][l] = y_pred[x][l] / sum

        for l in range(len(model['W2'])):
            loss=loss + (y_vector[x][l] * math.log(y_pred[x][l]))
        print(y_pred[x],y[x],y_vector[x],loss)
    loss = (-1)*loss / input_size
    print(loss)
    return model
