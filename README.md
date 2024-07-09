# Neural-Network-from-Scratch
<p align="center">
  <img src="https://github.com/aryanc381/Neural-Network-from-Scratch/blob/main/mnist.png" alt="MNIST referance" width="1400" height="300">
</p>

In this project, I have hardcoded a neural network to classify handwritten digits from 0 to 9 using the MNIST dataset.

## Libraries I used : 
1. NumPy
2. Pandas
3. Matplotlib

## Important note :
- Tensorflow is not required.
- Keras is not required.

## Pre-requisites :
1. Code-Editor : VSCode, Jupyter NB, Anaconda
2. Command-Line installation (copy) : pip install numpy, pandas, matplotlib

## Layout of the netowrk : 
1. **Input Layer :** Consists of the **'X_train.shape[0]'** neurons which is determined by the number of input features of the input data.
2. **First Hidden Layer :** There are 10 neurons, I have used ReLU (Rectified Linear Unit) as the activation function.
3. **Second Hidden Layer :** Again, there are 10 neurons and I have used ReLU again as an activation fucntion.
4. **Output Layer :** I have used the SoftMax function and hence the number of neurons are dynamically decided by the number of classes in **'Y_train'**.

## Code Explanation : 
**- def init_params() :**
1. We initialised w1 with a random matrix for X_train values using numpy's randn function.
2. Here X_train.shape[0] depicts the number of features present in the training data X_train.
3. np.sqrt(1 / X_train.shape[0]) : This is Xavier initialization to scale randomly generated weights that helps to converge and attain global minimum.
4. b1, b2 : Creating bias vectors (10 rows, 1 column).

**- def ReLU(z) :**
1. **Input** - A Rectified Linear unit can take any form of input that can be a scalar value, a vector or a matrix (typically the result of the linear transformation).
2. **Operation** - If the element is greater than zero i.e (z > 0), the the result remains unchanged, but if the element is less than zero i.e (z < 0), then the result returns zero (0).

<p align="center">
  <img src="https://github.com/aryanc381/Neural-Network-from-Scratch/blob/main/ReLU.png" alt="ReLU Activation Function" width="600" height="300">
</p>

3. **Function** - ReLU behaves like a linear function for z greater than zero and zero i.e non linear for the values of z less than zero.
4. **Why is it required** - When the negative values are ruled out by zeroing them, the network converges faster and becomes more efficient.
5. **Disadvantages** - For the dataset where negative values are important, the model does not learn from them hence ReLU must be used where only 0 - infinity values are important.

**- def softmax(z) :**
1. **Input** - The softmax function has the input of the linear transformation in a neural network.
2. **Exponentiation np.exp(z - np.max(z))** - Subtracting the **np.max(z)** ensures that the largest value z becomes zero after the subtraction., this helps in the adjustment in reducing the range of exponentiation.
3. **Normalisation exp_z / np.sum(exp_z, axis=0, keepdims=True)** - Computes the sum of the **'exp_z'** along the zeroth axis ensuring that the softmax function value sums up to 1 across the output vector. Finally the normalisation takes place to ensure probabilities ensuring they represent a probability distribution.

<p align="center">
  <img src="softmax.png" alt="ReLU Activation Function" width="600" height="300">
</p>

4. **How it works** - The Softmax function converts the input to a probability distribution where each element in the output vector represents probability of the corresponding class.
5. **Output Range** - The output range is from 0 - 1 making it suitable for multi-class classification tasks.
6. **Usage** - Softmax function is used in backprop.

**- def forward_prop(w1, b1, w2, b2, x) :**
1. **z1, z2** - Linear Activation function : w.x + b
2. **a1** - ReLU(z1)
3. **a2** - Softmax(z2)

**- def one_hot(y) :**
1. **Functionality** - Converts a categorical array into its one-hot encoded representation.
2. **Initialization: np.zeros((y.size, y.max() + 1))** - Creates a matrix one_hot_y of zeros with dimensions (y.size, y.max() + 1), here y is the number of elements and y.max() + 1 is the number of columns represents maximum value of y plus one.
3. **Setting Indices: one_hot_y[np.arange(y.size), y] = 1** - Creates an array of indices from 0 to y.size-1, y is used as indices along the second axis of one_hot_y.
4. **Transpose: one_hot_y = one_hot_y.T** - Transposes one_hot_y to make the one-hot encoded vectors (1s and 0s) aligned as rows, with each row representing a class and each column representing an instance.
5. **Binary Representation** - One-hot encoding converts categorical variables into a binary matrix where each column corresponds to a unique category and each row represents an instance.
6. **Usage** - This representation is commonly used in machine learning tasks, especially for multi-class classification where algorithms require inputs in numerical format.

**- def deriv_ReLU(z) :**
1. **Derivative** - returns z when z > 0.

**- def def back_prop(z1, a1, z2, a2, w2, x, y) :**
1. **Function** - Computes the gradient descent in the neural network by the weights and biases created by forward prop.
2. **Error Propagation** - Backpropagation propagates the error from the output layer back through the network to compute gradients for each layer's weights and biases.
3. **Gradient Calculation** - Using the chain rule, it calculates how changes in weights and biases affect the loss function, allowing the network to update its parameters to minimize the loss.

**def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha) :**
1. **Function** - With the help of gradient descent, updating the parameters using alpha rate.

**def get_predictions(a2) :**
1. **Function** - Getting the predictions using the highest probability available in the zeroth axis.

**def get_accuracy(predictions, y) :**
1. **Function** - Determines the accuracy of the function by comparing the actual value 'y' with the predicted value 'y_hat'.

**def gradient_descent(x, y, iterations, alpha) :**
1. **Function** - Computing the gradient descent to achieve the global minimum.
2. **Delta W** - (w - alpha*dw/dz) : for weights.
3. **Delta b** - (b - alpha*db/dz) : for bias.
4. **Computation** - For every 50 iterations, the accuracy is printed for conclusion. (In code, 1000 iterations give 91.83% accuracy to the model). In tensorflow, these are epochs.
5. **Workflow** - Forward_prop() -> Back_prop() -> update_params().

**- def test_prediction(index, w1, b1, w2, b2) :**
1. **Display** - After the predictions have been made, the label and the predictions are displayed using this function.
2. **Plot** - The last 4 lines of code use matplotlib to show the handwritten image to cross-verify with the label.

## References : 
1. [Concept](https://www.youtube.com/watch?v=aircAruvnKk&t=812s&ab_channel=3Blue1Brown)
2. [Math behind ML and Neural Networks](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH)
