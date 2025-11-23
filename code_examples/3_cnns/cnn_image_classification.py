"""
Convolutional Neural Networks (CNNs): Image Classification
Complete implementation for image classification using CNNs
Demonstrates: Architecture, Training, Evaluation, and Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# EXAMPLE 1: CNN ARCHITECTURE EXPLANATION
# ============================================================================
def explain_cnn_architecture():
    """
    Explain CNN building blocks and architecture
    """
    print("=" * 80)
    print("CNN ARCHITECTURE EXPLANATION")
    print("=" * 80)
    
    architecture = {
        "Convolutional Layer": {
            "Purpose": "Extract local features (edges, corners, patterns)",
            "Operation": "Slide filters over image, compute element-wise multiplication",
            "Hyperparameters": {
                "Filters/Kernels": "Number of feature maps (e.g., 32, 64, 128)",
                "Kernel Size": "Usually 3x3 or 5x5",
                "Stride": "Step size when sliding filter (usually 1 or 2)",
                "Padding": "Add zeros around image (same or valid)"
            },
            "Output": "Feature maps (e.g., 28x28x32 for 32 filters)"
        },
        
        "Activation Function": {
            "Purpose": "Add non-linearity to enable learning complex patterns",
            "Common": "ReLU (Rectified Linear Unit): max(0, x)",
            "Alternatives": "Sigmoid, Tanh, LeakyReLU, ELU",
            "Why Needed": "Without activation, network is just linear transformation"
        },
        
        "Pooling Layer": {
            "Purpose": "Reduce spatial dimensions, keep important features",
            "Max Pooling": "Take maximum value in window (most common)",
            "Average Pooling": "Take average value in window",
            "Benefits": [
                "Reduces computation",
                "Makes features translation-invariant",
                "Prevents overfitting"
            ],
            "Common Size": "2x2 with stride 2"
        },
        
        "Fully Connected Layer": {
            "Purpose": "Classification based on learned features",
            "Location": "At the end of network",
            "Function": "Flatten feature maps → Dense layer → Output (classes)"
        },
        
        "Dropout": {
            "Purpose": "Regularization to prevent overfitting",
            "Mechanism": "Randomly deactivate neurons during training",
            "Rate": "Usually 0.5 (50% dropout)"
        }
    }
    
    for component, details in architecture.items():
        print(f"\n{component}:")
        print("-" * 60)
        for key, value in details.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    • {k}: {v}")
            elif isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    • {item}")
            else:
                print(f"  {key}: {value}")


# ============================================================================
# EXAMPLE 2: SIMPLE CNN FROM SCRATCH (WITHOUT FRAMEWORKS)
# ============================================================================
def convolution_operation(image: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """
    Perform 2D convolution operation
    
    Args:
        image: Input image (H x W)
        kernel: Filter kernel (K x K)
        padding: Zero padding size
    
    Returns:
        Convolved output
    """
    # Add padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    # Output dimensions
    H, W = image.shape
    K = kernel.shape[0]
    output_H, output_W = H - K + 1, W - K + 1
    
    output = np.zeros((output_H, output_W))
    
    # Slide kernel over image
    for i in range(output_H):
        for j in range(output_W):
            # Extract patch
            patch = image[i:i+K, j:j+K]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(patch * kernel)
    
    return output


def demonstrate_convolution():
    """
    Demonstrate convolution operation with simple example
    """
    print("\n" + "=" * 80)
    print("CONVOLUTION OPERATION DEMONSTRATION")
    print("=" * 80)
    
    # Create simple image
    image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=float)
    
    # Edge detection kernel
    kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float)
    
    print("\nInput Image (4x4):")
    print(image)
    print("\nSobel Edge Detection Kernel (3x3):")
    print(kernel)
    
    # Perform convolution
    output = convolution_operation(image, kernel, padding=1)
    
    print("\nOutput (Feature Map):")
    print(output)
    print(f"\nOutput Shape: {output.shape}")


# ============================================================================
# EXAMPLE 3: CNN WITH TENSORFLOW/KERAS
# ============================================================================
def cnn_with_tensorflow():
    """
    Build and train CNN using TensorFlow/Keras
    """
    print("\n" + "=" * 80)
    print("CNN WITH TENSORFLOW/KERAS")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, models
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.utils import to_categorical
        
        print("\n✓ TensorFlow imported successfully")
        print(f"  Version: {tf.__version__}")
        
        # Load MNIST dataset
        print("\nLoading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Normalize
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Reshape for CNN (add channel dimension)
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        
        # Convert labels to one-hot
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        print(f"  Training data: {X_train.shape}")
        print(f"  Test data: {X_test.shape}")
        
        # Build CNN model
        print("\nBuilding CNN model...")
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=(28, 28, 1), name='conv1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            
            # Flatten and Dense layers
            layers.Flatten(name='flatten'),
            layers.Dense(256, activation='relu', name='dense1'),
            layers.Dropout(0.5, name='dropout'),
            layers.Dense(10, activation='softmax', name='output')
        ])
        
        # Print model architecture
        print("\nModel Architecture:")
        model.summary()
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train on small subset (for demo)
        print("\nTraining model (on small subset for demo)...")
        history = model.fit(
            X_train[:5000], y_train[:5000],
            epochs=3,
            batch_size=128,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate
        print("\nEvaluating on test set...")
        test_loss, test_acc = model.evaluate(X_test[:1000], y_test[:1000], verbose=0)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = model.predict(X_test[:5], verbose=0)
        for i in range(5):
            pred_class = np.argmax(predictions[i])
            true_class = np.argmax(y_test[i])
            confidence = predictions[i][pred_class]
            print(f"  Image {i}: Predicted={pred_class}, True={true_class}, Confidence={confidence:.2%}")
        
        return model, X_test, y_test
        
    except ImportError:
        print("⚠ TensorFlow not installed. Install with: pip install tensorflow")
        return None, None, None


# ============================================================================
# EXAMPLE 4: FAMOUS CNN ARCHITECTURES
# ============================================================================
def famous_architectures():
    """
    Explain famous CNN architectures
    """
    print("\n" + "=" * 80)
    print("FAMOUS CNN ARCHITECTURES")
    print("=" * 80)
    
    architectures = {
        "LeNet (1998)": {
            "Designer": "Yann LeCun",
            "Purpose": "Handwritten digit recognition",
            "Key Feature": "First successful CNN",
            "Layers": "Conv → Pool → Conv → Pool → FC → FC",
            "Accuracy on MNIST": "99.2%",
            "Modern Use": "Educational, IoT devices"
        },
        
        "AlexNet (2012)": {
            "Designer": "Krizhevsky, Sutskever, Hinton",
            "Purpose": "ImageNet competition winner",
            "Key Feature": "Deep network (8 layers), GPU training, ReLU activation",
            "Breakthrough": "Reduced ImageNet error from 26% to 15%",
            "Impact": "Started deep learning revolution",
            "Innovations": [
                "GPU acceleration (NVIDIA)",
                "ReLU activation",
                "Dropout regularization",
                "Data augmentation"
            ]
        },
        
        "VGGNet (2014)": {
            "Designer": "Simonyan & Zisserman",
            "Key Feature": "Very deep (16-19 layers), small 3x3 filters",
            "Advantage": "Simpler architecture, easy to understand",
            "Disadvantage": "More parameters, slower training",
            "Variants": "VGG16, VGG19",
            "ImageNet Accuracy": "92.7%"
        },
        
        "ResNet (2015)": {
            "Designer": "He et al. (Microsoft)",
            "Key Innovation": "Residual connections (skip connections)",
            "Benefit": "Allows training very deep networks (152+ layers)",
            "ImageNet Accuracy": "96.4% (ResNet-152)",
            "Why Important": "Solved vanishing gradient problem",
            "Use Cases": "State-of-the-art classification, object detection"
        },
        
        "Inception/GoogLeNet (2014)": {
            "Designer": "Szegedy et al. (Google)",
            "Key Feature": "Inception modules (parallel convolutions)",
            "Advantage": "Multiple filter sizes in parallel",
            "Parameters": "Fewer than VGG despite better accuracy",
            "ImageNet Accuracy": "93.3%"
        },
        
        "MobileNet (2017)": {
            "Designer": "Google",
            "Purpose": "Efficient CNNs for mobile/edge devices",
            "Key Technique": "Depthwise Separable Convolution",
            "Advantage": "Small model size, fast inference",
            "Use": "Mobile apps, IoT, embedded systems",
            "Model Size": "17 MB (vs 100+ MB for ResNet)"
        }
    }
    
    for arch_name, details in architectures.items():
        print(f"\n{arch_name}:")
        print("-" * 70)
        for key, value in details.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    • {item}")
            else:
                print(f"  {key}: {value}")


# ============================================================================
# EXAMPLE 5: TRANSFER LEARNING
# ============================================================================
def transfer_learning_explanation():
    """
    Explain transfer learning with CNNs
    """
    print("\n" + "=" * 80)
    print("TRANSFER LEARNING WITH CNNs")
    print("=" * 80)
    
    concept = """
WHAT IS TRANSFER LEARNING?
Instead of training a CNN from scratch, we:
1. Use a pre-trained model (trained on ImageNet with 1M+ images)
2. Remove the final classification layer
3. Add new layers for our specific task
4. Fine-tune the model on our smaller dataset

WHY IS THIS USEFUL?
✓ Faster training (model already learned general features)
✓ Better accuracy (even with small datasets)
✓ Requires less computational power
✓ Reduces overfitting risk

COMMON PRE-TRAINED MODELS:
• ResNet50, ResNet152 - Very popular, good balance
• VGG16, VGG19 - Simple architecture
• Inception-v3 - Efficient
• MobileNetV2 - For mobile/edge devices
• EfficientNet - State-of-the-art efficiency

WORKFLOW:
1. Load pre-trained model (from ImageNet)
2. Freeze early layers (they learned general features)
3. Fine-tune last few layers (task-specific)
4. Train on your custom dataset

EXAMPLE CODE (using TensorFlow):

    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras import layers, models
    
    # Load pre-trained ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False)
    
    # Freeze early layers
    base_model.trainable = False
    
    # Add custom layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(custom_dataset, epochs=10)
"""
    
    print(concept)


# ============================================================================
# EXAMPLE 6: VISUALIZATION & INTERPRETATION
# ============================================================================
def visualize_kernels():
    """
    Visualize CNN filters/kernels
    """
    print("\n" + "=" * 80)
    print("CNN FILTER VISUALIZATION")
    print("=" * 80)
    
    # Common edge detection and feature filters
    filters = {
        "Horizontal Edge": np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]),
        
        "Vertical Edge": np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]),
        
        "Blur": np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]) / 9,
        
        "Sharpen": np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]),
        
        "Emboss": np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])
    }
    
    print("\nCommon CNN Filters:")
    print("-" * 70)
    for filter_name, kernel in filters.items():
        print(f"\n{filter_name}:")
        print(kernel)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "🎯" * 40)
    print("CONVOLUTIONAL NEURAL NETWORKS (CNNs)")
    print("Complete Guide: Architecture, Training, and Applications")
    print("🎯" * 40)
    
    # Run demonstrations
    explain_cnn_architecture()
    demonstrate_convolution()
    visualize_kernels()
    famous_architectures()
    transfer_learning_explanation()
    
    # Try TensorFlow if available
    model, X_test, y_test = cnn_with_tensorflow()
    
    print("\n" + "=" * 80)
    print("CNN TUTORIAL COMPLETE!")
    print("=" * 80)
    print("\n📚 KEY TAKEAWAYS:")
    print("  ✓ CNNs extract local features through convolution")
    print("  ✓ Pooling reduces dimensions while preserving features")
    print("  ✓ Deep architectures learn hierarchical representations")
    print("  ✓ Transfer learning accelerates training on custom tasks")
    print("  ✓ Modern CNNs achieve >99% accuracy on many tasks")
    print("\n🚀 NEXT STEPS:")
    print("  1. Study ResNet and skip connections")
    print("  2. Explore object detection (YOLO, R-CNN)")
    print("  3. Learn semantic segmentation")
    print("  4. Try transfer learning on custom dataset")
    print("\n" + "=" * 80)

