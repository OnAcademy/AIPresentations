# Section 1: Fundamentals of AI & Machine Learning

## Concepts Covered

1. **Supervised Learning**
   - Classification (Iris, Cats vs Dogs)
   - Regression (Housing Price Prediction)
   - Train/Test Split
   - Evaluation Metrics (Accuracy, Precision, Recall, F1)

2. **Unsupervised Learning**
   - Clustering (K-Means)
   - Dimensionality Reduction (PCA)
   - Anomaly Detection

3. **Reinforcement Learning Basics**
   - Agent-Environment Interaction
   - Rewards & Penalties
   - Q-Learning

## Files in This Section

- `supervised_learning.py` - Classification and Regression examples
- `unsupervised_learning.py` - Clustering and PCA
- `evaluation_metrics.py` - How to evaluate models
- `train_test_split.py` - Data splitting techniques
- `simple_nn.py` - Building a basic neural network from scratch

## Quick Examples

### Supervised Learning - Classification
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
```

### Unsupervised Learning - Clustering
```python
from sklearn.cluster import KMeans
import numpy as np

# Generate data
X = np.random.randn(300, 2)

# Cluster
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

print(f"Cluster centers:\n{kmeans.cluster_centers_}")
```

## Learning Resources

- Supervised vs Unsupervised: What's the difference?
- How evaluation metrics work
- Cross-validation techniques
- Bias-Variance Tradeoff

## Next Steps

After mastering fundamentals:
1. Move to Deep Learning (neural networks)
2. Explore specialized architectures (CNN, RNN)
3. Learn advanced techniques (transfer learning, fine-tuning)

