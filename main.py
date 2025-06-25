import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
from pathlib import Path
import cloudpickle
import time
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "olive_fly_model.cloudpkl"

#              Hyperparameters                #
###############################################
# iterations = 1750 # Change this value to change training length
# learning_rate = 0.00045
# hidden_size = 256 # Pay attention to changing this value (64, 256) need to be tested
iterations = 600 # Change this value to change training length
learning_rate = 0.0007
hidden_size = 256 # Pay attention to changing this value (64, 256) need to be tested
###############################################


def data_loader(dataset_path="big_ass_dataset", img_size=(64,64)):
    X = []
    y = []

    dataset_path = Path(dataset_path)

    #Get subdirs
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

                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img = img.resize(img_size)

                img_array = np.array(img, dtype=np.float32) / 255.0

                X.append(img_array)
                y.append(class_to_label[class_name])
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    return X, y, class_to_label

X, y, class_mapping = data_loader()

indices = np.random.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]

m = len(X_shuffled)
split_point = int(0.8 * m)

print(f"Total samples: {m}")
print(f"Training samples: {split_point}")
print(f"Validation samples: {m - split_point}")

X_train = X_shuffled[:split_point]
X_train_flat = X_train.reshape(X_train.shape[0], -1).T
Y_train = y_shuffled[:split_point]

X_val = X_shuffled[split_point:]
X_val_flat = X_val.reshape(X_val.shape[0], -1).T
Y_val = y_shuffled[split_point:]

def init_params():
    global hidden_size
    input_size = 64 * 64 * 3
    output_size = 2

    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((hidden_size, 1))

    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((output_size, 1))

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X, training=True, dropout_rate=0.3):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    if training:
        dropout_mask = np.random.rand(*A1.shape) > dropout_rate
        A1 = A1 * dropout_mask / (1 - dropout_rate)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1 
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y

    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)

    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def weighted_back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)

    class_0_weight = 0.3  # Not olive fly
    class_1_weight = 3.0  # Olive fly 
    
    # Apply weights to loss
    weights = np.where(Y == 0, class_0_weight, class_1_weight)
    weights = weights.reshape(1, -1)
    
    dZ2 = (A2 - one_hot_Y) * weights
    
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# def gradient_descent(X, Y, iterations, alpha):
#     W1, b1, W2, b2 = init_params()
#     for i in range(iterations):
#         Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
#         dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
#         # print(f"W1: {W1.shape} | dW1: {dW1.shape}")
#         # print(f"W2: {W2.shape} | dW2: {dW2.shape}")
#         W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
#         if i % 25 == 0:
#             print(f"Iteration: {i}")
#             print(f"Accuracy: {get_accuracy(get_predictions((A2)), Y)}")
#     return W1, b1, W2, b2

def gradient_descent(X, Y, iterations, alpha, batch_size=256):
    W1, b1, W2, b2 = init_params()
    m = X.shape[1]
    
    for i in range(iterations):
        indices = np.random.permutation(m)
        for j in range(0, m, batch_size):
            batch_indices = indices[j:j+batch_size]
            X_batch = X[:, batch_indices]
            Y_batch = Y[batch_indices]
            
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = weighted_back_prop(Z1, A1, Z2, A2, W2, X_batch, Y_batch) 
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 25 == 0:
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X, training=False)  
            print(f"Iteration: {i}")
            print(f"Accuracy: {get_accuracy(get_predictions(A2), Y)}")
    
    return W1, b1, W2, b2

class OliveFlyModel:
    def __init__(self, W1, b1, W2, b2, class_mapping):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.class_mapping = class_mapping

    def predict(self, X, threshold=0.3):
        if X.shape[1] == 4096: #Grayscale
            X_rgb = np.repeat(X.reshape(-1, 64, 64, 1), 3, axis=3)
            X = X_rgb.reshape(X.shape[0], -1)
        
        Z1, A1, Z2, A2 = forward_prop(self.W1, self.b1, self.W2, self.b2, X.T)
        olive_fly_probs = A2[1, :] # Probability of olive fly
        predictions = (olive_fly_probs > threshold).astype(int)
        # predictions = get_predictions(A2)
        return predictions

if __name__ == "__main__":
    #Training
    print("Starting training")
    start_time = time.time()    
    W1, b1, W2, b2 = gradient_descent(X_train_flat, Y_train, iterations, learning_rate)

    #Validation
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_val_flat)
    val_predictions = get_predictions(A2)
    val_accuracy = get_accuracy(val_predictions, Y_val)
    print(f"Validation accuracy: {val_accuracy}")
    end_time = time.time()
    total_time_hrs = (end_time - start_time) // 60
    print(f"Total time: {end_time - start_time}")

    model = OliveFlyModel(W1, b1, W2, b2, class_mapping)

    with open(MODEL_PATH, 'wb') as f:
        cloudpickle.dump(model, f)