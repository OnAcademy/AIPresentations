"""
Deep Learning: MNIST Classification with TensorFlow
Building and training a neural network from scratch
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)


# ============================================================================
# EXAMPLE 1: SIMPLE NEURAL NETWORK
# ============================================================================
def simple_neural_network():
    """
    Build and train a simple 3-layer neural network for MNIST
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simple Neural Network for MNIST")
    print("=" * 70)
    
    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Normalize pixel values from [0, 255] to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    print(f"Normalized data range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    
    # Build model (Sequential API)
    print("\nBuilding model...")
    model = tf.keras.Sequential([
        # Input: 28x28 image → Flatten to 784 values
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        
        # Hidden layer 1: 128 neurons, ReLU activation
        tf.keras.layers.Dense(128, activation='relu', name='hidden1'),
        
        # Dropout: randomly disable 20% of neurons (prevents overfitting)
        tf.keras.layers.Dropout(0.2),
        
        # Hidden layer 2: 64 neurons, ReLU activation
        tf.keras.layers.Dense(64, activation='relu', name='hidden2'),
        
        # Dropout: 20%
        tf.keras.layers.Dropout(0.2),
        
        # Output layer: 10 neurons (one per digit 0-9), Softmax
        tf.keras.layers.Dense(10, activation='softmax', name='output')
    ])
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Compile model
    print("\nCompiling model...")
    model.compile(
        optimizer='adam',  # Adam optimizer (adaptive learning rate)
        loss='sparse_categorical_crossentropy',  # For integer labels
        metrics=['accuracy']  # Track accuracy
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,  # Number of passes through entire dataset
        batch_size=32,  # Number of samples per gradient update
        validation_split=0.1,  # Use 10% of training data for validation
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, (X_train, y_train), (X_test, y_test), history


# ============================================================================
# EXAMPLE 2: CONVOLUTIONAL NEURAL NETWORK
# ============================================================================
def convolutional_neural_network():
    """
    Build a CNN for MNIST (better for image data)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Convolutional Neural Network (CNN)")
    print("=" * 70)
    
    # Load and normalize data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape for CNN: (samples, height, width, channels)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    print(f"Reshaped training data: {X_train.shape}")
    
    # Build CNN model
    print("\nBuilding CNN model...")
    model = tf.keras.Sequential([
        # Convolutional layer 1: 32 filters, 3x3 kernel, ReLU
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                              input_shape=(28, 28, 1), name='conv1'),
        
        # Max pooling: reduces spatial dimensions
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Convolutional layer 2: 64 filters
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        
        # Max pooling
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Flatten for dense layers
        tf.keras.layers.Flatten(name='flatten'),
        
        # Dense layers
        tf.keras.layers.Dense(64, activation='relu', name='dense1'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax', name='output')
    ])
    
    print("\nCNN Architecture:")
    model.summary()
    
    # Compile and train
    print("\nCompiling and training CNN...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    return model, history


# ============================================================================
# EXAMPLE 3: MAKE PREDICTIONS
# ============================================================================
def make_predictions(model, X_test, y_test):
    """
    Make predictions on test data and visualize results
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Making Predictions")
    print("=" * 70)
    
    # Get predictions
    predictions = model.predict(X_test[:10])  # First 10 test samples
    predicted_labels = np.argmax(predictions, axis=1)
    
    print("\nSample Predictions:")
    print(f"{'Index':<6} {'Predicted':<12} {'Actual':<8} {'Correct':<8} {'Confidence'}")
    print("-" * 50)
    
    for i in range(10):
        pred = predicted_labels[i]
        actual = y_test[i]
        confidence = predictions[i][pred]
        correct = "✓" if pred == actual else "✗"
        
        print(f"{i:<6} {pred:<12} {actual:<8} {correct:<8} {confidence:.4f}")
    
    # Visualize predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('MNIST Predictions')
    
    for i, ax in enumerate(axes.flat):
        # Reshape data back to 28x28 for visualization
        image = X_test[i].reshape(28, 28) if X_test.ndim == 4 else X_test[i]
        
        ax.imshow(image, cmap='gray')
        pred = predicted_labels[i]
        actual = y_test[i]
        color = 'green' if pred == actual else 'red'
        ax.set_title(f'Pred: {pred} (Actual: {actual})', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=100, bbox_inches='tight')
    print("\n✓ Predictions visualization saved as 'mnist_predictions.png'")


# ============================================================================
# EXAMPLE 4: VISUALIZE TRAINING HISTORY
# ============================================================================
def plot_training_history(history):
    """
    Plot training and validation loss/accuracy
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Training History")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
    print("✓ Training history saved as 'training_history.png'")


# ============================================================================
# EXAMPLE 5: HYPERPARAMETER TUNING
# ============================================================================
def explain_hyperparameters():
    """
    Explain important hyperparameters and their effects
    """
    print("\n" + "=" * 70)
    print("KEY HYPERPARAMETERS")
    print("=" * 70)
    
    params = {
        "Epochs": {
            "Description": "Number of times to iterate over training data",
            "Too Low": "Model doesn't converge, poor performance",
            "Too High": "Wastes computation, may overfit",
            "Typical Range": "10-100"
        },
        
        "Batch Size": {
            "Description": "Number of samples processed before updating weights",
            "Small (8-32)": "Noisy updates, slower but sometimes better",
            "Large (256+)": "Stable updates, faster, may get stuck in local minima",
            "Typical Range": "32-128"
        },
        
        "Learning Rate": {
            "Description": "Step size for gradient descent",
            "Too High": "Diverges, never converges",
            "Too Low": "Converges very slowly",
            "Default": "0.001 (Adam)"
        },
        
        "Dropout": {
            "Description": "Randomly disable neurons during training",
            "Effect": "Prevents overfitting, improves generalization",
            "Typical Range": "0.2-0.5"
        },
        
        "Hidden Units": {
            "Description": "Number of neurons in hidden layers",
            "Too Few": "Underfitting, model too simple",
            "Too Many": "Overfitting, slow training",
            "Rule of Thumb": "Between input and output size"
        }
    }
    
    for param, details in params.items():
        print(f"\n{param}:")
        for key, value in details.items():
            print(f"  {key}: {value}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Run examples
    model1, (X_train, y_train), (X_test, y_test), history1 = simple_neural_network()
    plot_training_history(history1)
    make_predictions(model1, X_test, y_test)
    
    print("\n" + "=" * 70)
    print("SECOND EXAMPLE: CNN (Run separately for cleaner output)")
    print("=" * 70)
    print("Uncomment the line below to train the CNN model:")
    print("# model2, history2 = convolutional_neural_network()")
    
    explain_hyperparameters()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Neural networks learn patterns from data through backpropagation
2. Activation functions introduce non-linearity (ReLU is popular)
3. Dropout prevents overfitting by randomly disabling neurons
4. Always normalize/standardize input data
5. Monitor validation loss to detect overfitting
6. CNNs are specialized for image data (spatial structure)
7. Hyperparameter tuning is crucial for good performance
8. Start simple, then gradually increase complexity
    """)

