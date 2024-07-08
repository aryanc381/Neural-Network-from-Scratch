# libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading the csv
data = pd.read_csv('E:\\aryan\\projects\\data\\train.csv')

# creating dimensions in form of m x n
m, n = data.shape
np.random.shuffle(data.values) # shuffling the data values

# Splitting the data into development set and training set and transposing for better manipulation
data_dev = data.iloc[0:1000].values.T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

# Training set 
data_train = data.iloc[1000:m].values.T
Y_train = data_train[0]
X_train = data_train[1:n]

# Normalize the input data
X_train = X_train / 255.0
X_dev = X_dev / 255.0

# Function used to initialise the weights of the neural network, the variance in the neural netork is maintained which assures that the gradients do not vanish from training.
# Variance of the o/p = Variance of the i/p
def init_params():
    w1 = np.random.randn(10, X_train.shape[0]) * np.sqrt(1 / X_train.shape[0])  # Xavier initialization
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(10, 10) * np.sqrt(1 / 10)  # Xavier initialization
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

# ReLU Activation Layer
def ReLU(z):
    return np.maximum(0, z)

# Stable Softmax activation Layer
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Numerically stable softmax
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Forward-propogation to achieve the activations
def forward_prop(w1, b1, w2, b2, x):
    z1 = np.dot(w1, x) + b1
    a1 = ReLU(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

# Convert layers into one-hot encoding which is basically converting the categorical data into vectors of 0s and 1s that can be fed to the learning ML Algorithm, 1 - hot, 0 - cold
def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

# Derivation of the ReLU Activation function
def deriv_ReLU(z):
    return z > 0

# BAckward propogation to compute gradient descent values
def back_prop(z1, a1, z2, a2, w2, x, y):
    m = y.size
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * deriv_ReLU(z1)
    dw1 = (1/m) * np.dot(dz1, x.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
    return dw1, dw2, db1, db2

# Update the parameters using gradient descent with an alpha rate
def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    w2 = w2 - alpha * dw2
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

# Getting the predictions by choosing the index that has the highest probability 
def get_predictions(a2):
    return np.argmax(a2, axis=0)

# Calculating the accuracy by comparing the predictions with the true labels 
def get_accuracy(predictions, y):
    return np.mean(predictions == y) * 100

# Computing gradient descent on the dataset to achieve global minimum
def gradient_descent(x, y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, dw2, db1, db2 = back_prop(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 50 == 0:
            print("Iterations : ", i)
            predictions = get_predictions(a2)
            accuracy = get_accuracy(predictions, y)
            print("Accuracy   : ", accuracy)
    return w1, w2, b1, b2

# Training the model using gradient descent 
w1, w2, b1, b2 = gradient_descent(X_train, Y_train, 1000, 0.1)  # Increase iterations to 1000

def make_predictions(x, w1, b1, w2, b2):
    _, _, _, A2 = forward_prop(w1, b1, w2, b2, x)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, w1, b1, w2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], w1, b1, w2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, w1, b1, w2, b2)
test_prediction(1, w1, b1, w2, b2)
test_prediction(2, w1, b1, w2, b2)
test_prediction(3, w1, b1, w2, b2)