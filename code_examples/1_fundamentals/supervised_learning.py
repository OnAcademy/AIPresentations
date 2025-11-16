"""
Supervised Learning Examples
Classification and Regression demonstrations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# ============================================================================
# EXAMPLE 1: CLASSIFICATION - Iris Dataset
# ============================================================================
def classification_example():
    """
    Classify iris flowers into species (Setosa, Versicolor, Virginica)
    This is supervised learning because we have labeled data
    """
    print("=" * 70)
    print("CLASSIFICATION EXAMPLE: Iris Flower Classification")
    print("=" * 70)
    
    # Load dataset with labels
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"\nDataset Info:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]} - {feature_names}")
    print(f"  Classes: {len(target_names)} - {target_names}")
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain/Test Split:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")
    
    # Train a classifier
    print(f"\n{'Model':<20} {'Accuracy':<12}")
    print("-" * 32)
    
    # Method 1: Logistic Regression
    lr_model = LogisticRegression(max_iter=200)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"{'Logistic Regression':<20} {lr_acc:.4f}")
    
    # Method 2: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"{'Random Forest':<20} {rf_acc:.4f}")
    
    # Detailed metrics for best model
    print(f"\nDetailed Classification Report (Random Forest):")
    print(classification_report(y_test, rf_pred, target_names=target_names))
    
    # Make predictions on new data
    sample = iris.data[0:1]
    prediction = rf_model.predict(sample)
    confidence = rf_model.predict_proba(sample)
    
    print(f"\nSample Prediction:")
    print(f"  Input: {sample[0]}")
    print(f"  Predicted class: {target_names[prediction[0]]}")
    print(f"  Confidence: {confidence[0]}")
    
    return rf_model, X_test, y_test


# ============================================================================
# EXAMPLE 2: REGRESSION - Housing Price Prediction
# ============================================================================
def regression_example():
    """
    Predict house prices based on features (area, rooms, etc.)
    Continuous output instead of discrete classes
    """
    print("\n" + "=" * 70)
    print("REGRESSION EXAMPLE: House Price Prediction")
    print("=" * 70)
    
    # Generate synthetic housing data
    X, y = make_regression(n_samples=200, n_features=4, noise=10, random_state=42)
    feature_names = ["Area (sqft)", "Bedrooms", "Bathrooms", "Age (years)"]
    
    print(f"\nDataset Info:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]} - {feature_names}")
    print(f"  Output: Price (continuous)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train models
    print(f"\n{'Model':<25} {'R² Score':<12} {'RMSE':<12}")
    print("-" * 49)
    
    # Method 1: Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    print(f"{'Linear Regression':<25} {lr_r2:.4f}       {lr_rmse:.4f}")
    
    # Method 2: Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    print(f"{'Random Forest':<25} {rf_r2:.4f}       {rf_rmse:.4f}")
    
    # Make a prediction
    sample = X[0:1]
    price = rf_model.predict(sample)[0]
    
    print(f"\nSample Prediction:")
    print(f"  Input: Area={sample[0,0]:.0f} sqft, Beds={sample[0,1]:.0f}, "
          f"Baths={sample[0,2]:.0f}, Age={sample[0,3]:.0f} years")
    print(f"  Predicted Price: ${price:.2f}")
    
    # Feature importance
    print(f"\nFeature Importance (Random Forest):")
    for name, importance in zip(feature_names, rf_model.feature_importances_):
        print(f"  {name:<20} {importance:.4f}")
    
    return rf_model, X_test, y_test


# ============================================================================
# EXAMPLE 3: KEY CONCEPTS
# ============================================================================
def explain_key_concepts():
    """
    Explain the fundamental concepts of supervised learning
    """
    print("\n" + "=" * 70)
    print("KEY CONCEPTS IN SUPERVISED LEARNING")
    print("=" * 70)
    
    concepts = {
        "Training Data": 
            "Labeled examples used to train the model. "
            "The model learns patterns from these examples.",
        
        "Test Data": 
            "Held-out data used to evaluate model performance. "
            "Must not be used during training!",
        
        "Overfitting": 
            "Model memorizes training data (low train error, high test error). "
            "Solution: Regularization, more data, simpler model.",
        
        "Underfitting": 
            "Model is too simple to capture patterns (high train & test error). "
            "Solution: More complex model, more features.",
        
        "Accuracy": 
            "Percentage of correct predictions. Good for balanced datasets.",
        
        "Precision": 
            "Of predicted positives, how many are actually positive? "
            "Important for spam detection.",
        
        "Recall": 
            "Of actual positives, how many did we find? "
            "Important for disease detection.",
        
        "F1 Score": 
            "Harmonic mean of Precision and Recall. "
            "Good metric for imbalanced datasets.",
    }
    
    for concept, explanation in concepts.items():
        print(f"\n{concept}:")
        print(f"  {explanation}")


# ============================================================================
# EXAMPLE 4: VISUALIZATION
# ============================================================================
def visualize_results(model, X_test, y_test, title="Model Performance"):
    """
    Visualize model predictions vs actual values
    """
    try:
        predictions = model.predict(X_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Predictions vs Actual
        axes[0].scatter(y_test, predictions, alpha=0.6, s=50)
        axes[0].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel("Actual Values")
        axes[0].set_ylabel("Predicted Values")
        axes[0].set_title(f"{title}: Predictions vs Actual")
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_test - predictions
        axes[1].scatter(predictions, residuals, alpha=0.6, s=50)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel("Predicted Values")
        axes[1].set_ylabel("Residuals")
        axes[1].set_title("Residual Plot")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=100, bbox_inches='tight')
        print(f"\n✓ Visualization saved as 'model_performance.png'")
        
    except Exception as e:
        print(f"Could not visualize (probably classification data): {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Run all examples
    classification_example()
    regression_example()
    explain_key_concepts()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Supervised Learning needs labeled data
2. Split data: training (learn patterns) + testing (evaluate)
3. Classification: predict discrete categories
4. Regression: predict continuous values
5. Always evaluate on test data, never on training data!
6. Choose appropriate metrics for your problem
7. Watch out for overfitting and underfitting
    """)

