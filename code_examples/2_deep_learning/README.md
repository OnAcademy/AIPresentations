# Section 2: Deep Learning Fundamentals

## Concepts Covered

1. **Neural Network Basics**
   - Perceptrons & Layers
   - Forward Propagation
   - Activation Functions (ReLU, Sigmoid, Tanh)

2. **Backpropagation**
   - Gradient Descent
   - Loss Functions
   - Optimization Algorithms (SGD, Adam)

3. **Training Neural Networks**
   - Building networks with TensorFlow/PyTorch
   - Loss and Metrics
   - Epochs and Batch Size

4. **Overfitting Prevention**
   - Regularization (L1, L2, Dropout)
   - Early Stopping
   - Cross-Validation

## Files in This Section

- `neural_network_basics.py` - From scratch NN implementation
- `tensorflow_mnist.py` - MNIST classification with TensorFlow
- `pytorch_mnist.py` - MNIST classification with PyTorch
- `activation_functions.py` - Visualizing activation functions
- `regularization.py` - Dropout, L1/L2 regularization

## Neural Network Architecture

```
Input Layer (784 neurons for MNIST)
    ↓
Hidden Layer 1 (128 neurons, ReLU)
    ↓
Hidden Layer 2 (64 neurons, ReLU)
    ↓
Output Layer (10 neurons, Softmax)
```

## Quick Example: MNIST with TensorFlow

```python
import tensorflow as tf

# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

## Key Concepts

### Activation Functions
- **ReLU**: max(0, x) - Most popular, avoids vanishing gradient
- **Sigmoid**: 1/(1+e^-x) - Classic, prone to vanishing gradient
- **Tanh**: (e^x - e^-x)/(e^x + e^-x) - Similar to sigmoid, zero-centered

### Loss Functions
- **MSE**: For regression
- **Cross-Entropy**: For classification
- **Sparse Categorical CrossEntropy**: For multi-class classification

### Optimizers
- **SGD**: Simple, but slow
- **Adam**: Adaptive Learning Rate, popular choice
- **RMSprop**: Good for RNNs

## Common Mistakes

❌ Not normalizing input data
❌ Learning rate too high (diverges) or too low (slow convergence)
❌ Overfitting - using too complex model on small dataset
❌ Using activation functions before understanding them
❌ Training on entire dataset (no validation/test split)

## Metrics to Track

- **Training Loss**: Should decrease over epochs
- **Validation Loss**: Should follow training loss (if diverges = overfitting)
- **Accuracy**: Percentage of correct predictions
- **Precision/Recall**: For imbalanced classification

## Next Steps

1. Experiment with different architectures
2. Try different activation functions and optimizers
3. Learn about regularization techniques
4. Move to specialized architectures (CNN, RNN)

