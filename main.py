import numpy as np
import os
import glob
from PIL import Image
from pathlib import Path
import cloudpickle
from layers import ConvolutionLayer, MaxPoolLayer, FlattenLayer, DenseLayer, DropoutLayer

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "olive_fly_model_1.cloudpkl"

#              Hyperparameters                #
###############################################
iterations = 2000
learning_rate = 0.001
cnn_learning_rate = 0.0001
hidden_size = 512
dropout_rate = 0.75
###############################################

# CNN layers
conv1 = ConvolutionLayer(num_filters=32, filter_size=5, stride=1, padding=2)
pool1 = MaxPoolLayer(pool_size=(2, 2), stride=2)
dropout1 = DropoutLayer(keep_prob=dropout_rate)

conv2 = ConvolutionLayer(num_filters=64, filter_size=3, stride=1, padding=1)
pool2 = MaxPoolLayer(pool_size=(2, 2), stride=2)
dropout2 = DropoutLayer(keep_prob=dropout_rate)

flatten = FlattenLayer()

# Dense layers
input_size = 16 * 16 * 64  # After conv layers: 16x16x64
dense1 = DenseLayer(input_size, hidden_size, activation='relu')
dense2 = DenseLayer(hidden_size, 2, activation='softmax')

def data_loader(dataset_path="dataset", img_size=(64,64)):
    """Load and preprocess dataset"""
    X = []
    y = []

    dataset_path = Path(dataset_path)

    class_dirs = [dir for dir in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, dir))]
    class_dirs.sort()
    print(f"Found classes: {class_dirs}")

    class_to_label = {class_name: idx for idx, class_name in enumerate(class_dirs)}
    print(f"Found labels: {class_to_label}")

    for class_name in class_dirs:
        class_path = os.path.join(dataset_path, class_name)

        jpg_files = glob.glob(os.path.join(class_path, "*.jpg"))
        jpg_files.extend(glob.glob(os.path.join(class_path, "*.JPG")))

        for img_path in jpg_files:
            try:
                img = Image.open(img_path)

                if img.mode != 'L':
                    img = img.convert('L')
                
                img = img.resize(img_size)

                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = img_array.reshape(64, 64, 1)

                X.append(img_array)
                y.append(class_to_label[class_name])
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    return X, y, class_to_label

def ReLU(Z):
    """ReLU activation function"""
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    """Derivative of ReLU"""
    return Z > 0

def one_hot(Y):
    """Convert labels to one-hot encoding"""
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1 
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def get_predictions(A2):
    """Get class predictions from output"""
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    """Calculate accuracy"""
    return np.sum(predictions == Y) / Y.size

def cnn_forward(img, training=True):
    """Forward pass through CNN layers"""
    x = conv1.forward(img)
    x = ReLU(x)
    x = pool1.forward(x)
    x = dropout1.forward(x, training=training)
    
    x = conv2.forward(x)
    x = ReLU(x)
    x = pool2.forward(x)
    x = dropout2.forward(x, training=training)
    
    x = flatten.forward(x)
    return x

def cnn_backward(dout):
    """Backward pass through CNN layers"""
    dx = flatten.backward(dout)
    
    dx = dropout2.backward(dx)
    dx = pool2.backward(dx)
    dx = dx * deriv_ReLU(dx)
    dx = conv2.backward(dx, cnn_learning_rate)
    
    dx = dropout1.backward(dx)
    dx = pool1.backward(dx)
    dx = dx * deriv_ReLU(dx)
    dx = conv1.backward(dx, cnn_learning_rate)
    
    return dx

def forward_cnn_batch(images, training=True):
    """Forward pass through CNN for batch of images"""
    batch_features = []
    
    for img in images:
        features = cnn_forward(img, training=training)
        batch_features.append(features)
    
    return np.array(batch_features).T

def network_forward(X_features):
    """Forward pass through dense layers using DenseLayer objects"""
    A1 = dense1.forward(X_features)
    A2 = dense2.forward(A1)
    return A1, A2

def network_backward(A1, A2, Y):
    """Backward pass through dense layers using DenseLayer objects"""
    # Convert labels to one-hot encoding
    one_hot_Y = one_hot(Y)
    
    # For softmax + cross-entropy, gradient is (A - Y)
    dA2 = A2 - one_hot_Y
    
    # Backward through dense layers
    dA1 = dense2.backward(dA2, learning_rate)
    dX = dense1.backward(dA1, learning_rate)
    
    return dX

def extract_features_batch(X_images):
    """Extract CNN features from batch of images"""
    print("Extracting CNN features...")
    return forward_cnn_batch(X_images, training=False)

def gradient_descent_features(X_features, Y, iterations):
    """Train dense layers on pre-extracted features"""
    for i in range(iterations):
        # Forward pass
        A1, A2 = network_forward(X_features)
        
        # Backward pass
        network_backward(A1, A2, Y)
        
        if i % 50 == 0:
            accuracy = get_accuracy(get_predictions(A2), Y)
            print(f"Iteration: {i}, Accuracy: {accuracy:.4f}")
    
    return dense1, dense2

def gradient_descent_end_to_end(X_images, Y, iterations):
    """End-to-end training of CNN + dense layers"""
    for i in range(iterations):
        # Forward pass through CNN (with dropout during training)
        X_features = forward_cnn_batch(X_images, training=True)
        
        # Forward through dense layers
        A1, A2 = network_forward(X_features)
        
        # Backward through dense layers
        dX_features = network_backward(A1, A2, Y)
        
        # Backward through CNN
        for j, img in enumerate(X_images):
            dout_flat = dX_features[:, j]
            cnn_backward(dout_flat)
        
        if i % 25 == 0:
            accuracy = get_accuracy(get_predictions(A2), Y)
            print(f"Iteration: {i}, Accuracy: {accuracy:.4f}")
    
    return dense1, dense2

class OliveFlyModel:
    """Model class for saving and loading"""
    def __init__(self, dense1, dense2, class_mapping):
        self.dense1 = dense1
        self.dense2 = dense2
        self.class_mapping = class_mapping

    def predict(self, X):
        """Make predictions on new data"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Convert flattened images to CNN format
        features = []
        for i in range(X.shape[0]):
            img = X[i].reshape(64, 64, 1)
            # Use training=False to disable dropout during prediction
            feature = cnn_forward(img, training=False)
            features.append(feature)
        
        X_features = np.array(features).T
        
        # Forward through dense layers
        A1, A2 = self.network_forward_predict(X_features)
        predictions = get_predictions(A2)
        return predictions
    
    def network_forward_predict(self, X_features):
        """Forward pass for prediction (using saved dense layers)"""
        A1 = self.dense1.forward(X_features)
        A2 = self.dense2.forward(A1)
        return A1, A2

# Load and prepare data
print("Loading dataset...")
X, y, class_mapping = data_loader(dataset_path=SCRIPT_DIR / "dataset")

# Shuffle and split data
indices = np.random.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]

m = len(X_shuffled)
split_point = int(0.8 * m)

print(f"Total samples: {m}")
print(f"Training samples: {split_point}")
print(f"Validation samples: {m - split_point}")

X_train = X_shuffled[:split_point]
Y_train = y_shuffled[:split_point]

X_val = X_shuffled[split_point:]
Y_val = y_shuffled[split_point:]

if __name__ == "__main__":
    print("\nChoose training approach:")
    print("1. Fast feature extraction (recommended)")
    print("2. End-to-end training (slower but potentially better)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\n=== Starting Feature Extraction Training ===")
        # Extract features once using CNN
        X_train_features = extract_features_batch(X_train)
        X_val_features = extract_features_batch(X_val)
        
        # Train dense layers on features
        trained_dense1, trained_dense2 = gradient_descent_features(X_train_features, Y_train, iterations)
        
        # Validate
        A1, A2 = network_forward(X_val_features)
        
    else:
        print("\n=== Starting End-to-End Training ===")
        # Train entire network end-to-end
        trained_dense1, trained_dense2 = gradient_descent_end_to_end(X_train, Y_train, 200)
        
        # Extract validation features for final evaluation
        X_val_features = extract_features_batch(X_val)
        A1, A2 = network_forward(X_val_features)

    # Final evaluation
    val_predictions = get_predictions(A2)
    val_accuracy = get_accuracy(val_predictions, Y_val)
    print(f"\n=== Final Results ===")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # Save model
    model = OliveFlyModel(trained_dense1, trained_dense2, class_mapping)

    with open(MODEL_PATH, 'wb') as f:
        cloudpickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")
    print("\nTraining complete!")