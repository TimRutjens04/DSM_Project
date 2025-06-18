import numpy as np
from pathlib import Path
import cloudpickle

# Import functions from main.py
from main import (
    data_loader, init_params, forward_prop, back_prop, update_params,
    get_predictions, get_accuracy, ReLU, softmax, deriv_ReLU, one_hot
)

SCRIPT_DIR = Path(__file__).resolve().parent
ENSEMBLE_MODEL_PATH = SCRIPT_DIR / "ensemble_olive_fly_models.cloudpkl"

#              Ensemble Hyperparameters       #
###############################################
num_models = 5
iterations = 2000
learning_rate = 0.0005
hidden_size = 256
###############################################

def gradient_descent_ensemble(X, Y, iterations, alpha, model_id):
    """Modified gradient descent that shows which model is training"""
    W1, b1, W2, b2 = init_params()
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 200 == 0:  # Less frequent printing for ensemble
            accuracy = get_accuracy(get_predictions(A2), Y)
            print(f"  Model {model_id} - Iteration: {i}, Accuracy: {accuracy:.4f}")
    
    return W1, b1, W2, b2

def train_ensemble_models(X, y, num_models=5):
    """Train multiple models with different random seeds and data splits"""
    models = []
    val_accuracies = []
    
    print(f"Training {num_models} models for ensemble...")
    print("=" * 60)
    
    for model_id in range(num_models):
        print(f"\n🤖 TRAINING MODEL {model_id + 1}/{num_models}")
        print("-" * 40)
        
        # Set different random seed for each model
        np.random.seed(42 + model_id * 17)
        
        # Create different train/val splits for each model
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # 80/20 split
        split_point = int(0.8 * len(X_shuffled))
        
        X_train_model = X_shuffled[:split_point]
        X_train_flat_model = X_train_model.reshape(X_train_model.shape[0], -1).T
        Y_train_model = y_shuffled[:split_point]
        
        X_val_model = X_shuffled[split_point:]
        X_val_flat_model = X_val_model.reshape(X_val_model.shape[0], -1).T
        Y_val_model = y_shuffled[split_point:]
        
        print(f"  Training samples: {len(Y_train_model)}")
        print(f"  Validation samples: {len(Y_val_model)}")
        
        # Train the model
        W1, b1, W2, b2 = gradient_descent_ensemble(
            X_train_flat_model, Y_train_model, iterations, learning_rate, model_id + 1
        )
        
        # Validate the model
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_val_flat_model)
        val_accuracy = get_accuracy(get_predictions(A2), Y_val_model)
        
        models.append({
            'W1': W1,
            'b1': b1, 
            'W2': W2,
            'b2': b2,
            'val_accuracy': val_accuracy,
            'model_id': model_id + 1
        })
        
        val_accuracies.append(val_accuracy)
        
        print(f"  ✅ Model {model_id + 1} validation accuracy: {val_accuracy:.4f}")
    
    return models, val_accuracies

def ensemble_predict(models, X_test, Y_test):
    """Make predictions using ensemble voting"""
    print("Making ensemble predictions...")
    
    all_predictions = []
    
    for i, model in enumerate(models):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_test)
        predictions = get_predictions(A2)
        all_predictions.append(predictions)
        
        print(f"  Model {i+1} individual accuracy on test: {get_accuracy(predictions, Y_test):.4f}")
    
    # Majority vote ensemble
    all_predictions = np.array(all_predictions)
    ensemble_predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)
    
    return ensemble_predictions, all_predictions

def save_ensemble(models, class_mapping, img_size=(64, 64)):
    """Save all ensemble models"""
    ensemble_data = {
        'models': models,
        'class_mapping': class_mapping,
        'img_size': img_size,
        'num_models': len(models),
        'individual_accuracies': [model['val_accuracy'] for model in models]
    }
    
    with open(ENSEMBLE_MODEL_PATH, 'wb') as f:
        cloudpickle.dump(ensemble_data, f)
    
    print(f"\n💾 Ensemble saved to: {ENSEMBLE_MODEL_PATH}")

def run_ensemble():
    """Main ensemble training pipeline"""
    print("🚀 ENSEMBLE TRAINING PIPELINE")
    print("=" * 60)
    
    # Load data
    print("📁 Loading dataset...")
    X, y, class_mapping = data_loader(dataset_path=SCRIPT_DIR / "dataset")
    print(f"Total samples: {len(X)}")
    print(f"Classes: {list(class_mapping.keys())}")
    
    # Train ensemble
    models, individual_accuracies = train_ensemble_models(X, y, num_models)
    
    # Test ensemble on a consistent validation set
    print(f"\n🧪 ENSEMBLE EVALUATION")
    print("=" * 40)
    
    # Create a consistent test split
    np.random.seed(42)  # Fixed seed for consistent evaluation
    indices = np.random.permutation(len(X))
    split_point = int(0.8 * len(X))
    
    X_val_final = X[indices[split_point:]]
    X_val_flat_final = X_val_final.reshape(X_val_final.shape[0], -1).T
    Y_val_final = y[indices[split_point:]]
    
    # Get ensemble predictions
    ensemble_pred, individual_preds = ensemble_predict(models, X_val_flat_final, Y_val_final)
    ensemble_accuracy = get_accuracy(ensemble_pred, Y_val_final)
    
    # Results summary
    print(f"\n📊 FINAL RESULTS")
    print("=" * 40)
    print(f"Individual model accuracies:")
    for i, acc in enumerate(individual_accuracies):
        print(f"  Model {i+1}: {acc:.4f}")
    print(f"Average individual accuracy: {np.mean(individual_accuracies):.4f}")
    print(f"Best individual model: {max(individual_accuracies):.4f}")
    print(f"🎯 ENSEMBLE ACCURACY: {ensemble_accuracy:.4f}")
    print(f"Improvement over best individual: {ensemble_accuracy - max(individual_accuracies):.4f}")
    
    # Save ensemble
    save_ensemble(models, class_mapping)
    
    return models, ensemble_accuracy

if __name__ == "__main__":
    # Set global hidden_size for imported functions
    import main
    main.hidden_size = hidden_size
    
    models, final_accuracy = run_ensemble()