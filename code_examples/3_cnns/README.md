# Section 3: Convolutional Neural Networks (CNNs)

## Concepts Covered

1. **CNN Architecture**
   - Convolutional layers (filters, kernels)
   - Pooling layers
   - Fully connected layers
   - Feature maps

2. **Key Components**
   - Filters and receptive fields
   - Stride and padding
   - Activation functions

3. **Classic Architectures**
   - LeNet (1998)
   - AlexNet (2012)
   - VGG (2014)
   - ResNet (2015)
   - EfficientNet (2019)

4. **Applications**
   - Image classification
   - Object detection (YOLO)
   - Semantic segmentation
   - Face recognition

5. **Transfer Learning**
   - Pre-trained models
   - Fine-tuning
   - Domain adaptation

## Key Advantages of CNNs

✓ **Parameter sharing**: Same filter used across entire image
✓ **Local connectivity**: Neurons only connect to small local regions
✓ **Hierarchical learning**: Learn simple features first, then complex
✓ **Translation invariance**: Detects features regardless of position

## Common CNN Architectures

| Model | Year | Accuracy | Parameters | Speed |
|-------|------|----------|-----------|-------|
| LeNet-5 | 1998 | 99.2% | 60K | Very fast |
| AlexNet | 2012 | 84.6% | 60M | Medium |
| VGG-16 | 2014 | 92.7% | 138M | Slow |
| ResNet-50 | 2015 | 93.1% | 25M | Fast |
| EfficientNet | 2019 | 94.4% | 5-66M | Very fast |

## Files in This Section

- `cnn_basics.py` - Building CNNs from scratch
- `image_classification.py` - CIFAR-10/ImageNet classification
- `transfer_learning.py` - Using pre-trained models
- `object_detection.py` - YOLO and R-CNN examples
- `visualization.py` - Understanding what CNNs learn

## CNN Process

```
Input Image (32x32x3)
    ↓
Conv Layer 1 (ReLU) → Feature Map 1
    ↓
Pool Layer 1 → Downsampled Feature Map
    ↓
Conv Layer 2 (ReLU) → Feature Map 2
    ↓
Pool Layer 2 → Downsampled Feature Map
    ↓
Flatten → 1D Vector
    ↓
Dense Layers → Classification Output (10 classes)
```

## Quick Example

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

## Learning Objectives

By the end of this section, you should understand:
- How convolutional layers work
- Why CNNs are superior for images
- Different CNN architectures
- How to use pre-trained models
- Transfer learning benefits
- Object detection basics

## Practical Applications

1. **Medical Imaging**: Tumor detection in X-rays
2. **Autonomous Vehicles**: Lane detection, obstacle avoidance
3. **Face Recognition**: Authentication, surveillance
4. **Quality Control**: Manufacturing defect detection
5. **Content Moderation**: Harmful image detection
6. **Recommendation Systems**: Visual similarity search

## Next Steps

1. Understand convolution operation
2. Build CNN from scratch with TensorFlow
3. Train on CIFAR-10 dataset
4. Implement transfer learning
5. Explore object detection
6. Visualize learned features

