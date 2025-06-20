import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, activation='relu'):
        """
        Dense (Fully Connected) Layer
        :param input_size - number of input features
        :param output_size - number of output neurons
        :param activation - activation function ('relu', 'softmax')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights using He initialization for ReLU
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((output_size, 1))
        
        # Cache for backpropagation
        self.last_input = None
        self.last_Z = None
        self.last_A = None
    
    def forward(self, X):
        """
        Forward pass through dense layer
        :param X - input tensor (features, batch_size) or (features,) for single sample
        :return - output after activation
        """
        # Handle single sample case
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        self.last_input = X
        
        # Linear transformation: Z = W*X + b
        self.last_Z = self.W.dot(X) + self.b
        
        # Apply activation function
        if self.activation == 'relu':
            self.last_A = self.relu(self.last_Z)
        elif self.activation == 'softmax':
            self.last_A = self.softmax(self.last_Z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        return self.last_A
    
    def backward(self, dA_next, learning_rate=0.001):
        """
        Backward pass through dense layer
        :param dA_next - gradient from next layer
        :param learning_rate - learning rate for weight updates
        :return - gradient w.r.t. input
        """
        m = self.last_input.shape[1]  # batch size
        
        # Apply activation derivative
        if self.activation == 'relu':
            dZ = dA_next * self.deriv_relu(self.last_Z)
        elif self.activation == 'softmax':
            # For softmax with cross-entropy, dZ is already computed as (A - Y)
            dZ = dA_next
        elif self.activation == 'linear':
            dZ = dA_next
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        # Compute gradients
        dW = (1 / m) * dZ.dot(self.last_input.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dX = self.W.T.dot(dZ)
        
        # Update weights
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return dX
    
    def relu(self, Z):
        """ReLU activation function"""
        return np.maximum(0, Z)
    
    def deriv_relu(self, Z):
        """Derivative of ReLU"""
        return Z > 0
    
    def softmax(self, Z):
        """Softmax activation function"""
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)
    
    def get_predictions(self, A):
        """Get class predictions from softmax output"""
        return np.argmax(A, 0)
    
    def get_accuracy(self, predictions, Y):
        """Calculate accuracy"""
        return np.sum(predictions == Y) / Y.size