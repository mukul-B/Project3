

def calculate_loss(model, X, y):
    print(model, X, y)
def predict(model, x):
    print(model, x)
    return x
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):


    # a = x*W1 + b1
    # h = tanh(a)
    # z = h*W2 + b2
    #
    print(X, y, nn_hdim, num_passes, print_loss)
    return X