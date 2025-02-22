import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
base_models = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('SVC', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)),
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('MLP', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
]

# Generate stacking features using out-of-fold predictions
def generate_stack_features(models, X_train, y_train):
    stack_features = []
    
    for name, model in models:
        if name == 'SVC':
            # For SVC, use decision function
            preds = cross_val_predict(model, X_train, y_train, cv=5, method='decision_function')
            if len(preds.shape) == 1:
                preds = preds.reshape(-1, 1)
        else:
            # For classifiers, use predict_proba
            preds = cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba')
        
        # If it's a binary classification, take only the positive class
        if preds.shape[1] == 2:
            preds = preds[:, 1]
        
        stack_features.append(preds)
    
    return np.hstack(stack_features)

# Generate stacked features for training
stack_train = generate_stack_features(base_models, X_train, y_train)

# Train meta-model (Logistic Regression)
meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(stack_train, y_train)

# Train base models on full training set
for name, model in base_models:
    model.fit(X_train, y_train)

# Generate stacking features for test set
stack_test = []
for name, model in base_models:
    if name == 'SVC':
        preds = model.decision_function(X_test)
        if len(preds.shape) == 1:
            preds = preds.reshape(-1, 1)
    else:
        preds = model.predict_proba(X_test)
    
    if preds.shape[1] == 2:
        preds = preds[:, 1]
    
    stack_test.append(preds)

stack_test = np.hstack(stack_test)

# Evaluate base models and ensemble
print("Model Accuracies:")
for name, model in base_models:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc:.4f}")

# Evaluate ensemble model
ensemble_preds = meta_model.predict(stack_test)
ensemble_acc = accuracy_score(y_test, ensemble_preds)
print(f"\nEnsemble Accuracy: {ensemble_acc:.4f}")