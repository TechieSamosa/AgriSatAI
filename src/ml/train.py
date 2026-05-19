import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.profiling import time_it
from config import settings

@time_it
def train_ensemble_classifier(df):
    """
    Trains an ensemble of models for crop health classification.
    Returns the best trained model and evaluation metrics.
    """
    print("Training Crop Health Classifier Ensemble...")
    
    # Feature selection
    features = ['mean_red', 'mean_nir', 'NDVI', 'temperature', 'soil_moisture', 'moisture_temp_interaction']
    target = 'status_label'
    
    # In a real scenario, we'd ensure 'status_label' is properly mapped.
    # Our synthetic data labels: healthy, stressed, critical
    
    X = df[features]
    y = df[target]
    
    # Generate some extra data if we have too few samples for CV
    if len(df) < 50:
        print("Not enough data, augmenting with synthetic noise...")
        augmented_X = [X]
        augmented_y = [y]
        for _ in range(20):
            noise = np.random.normal(0, 0.05, X.shape)
            augmented_X.append(X + noise)
            augmented_y.append(y)
        X = pd.concat(augmented_X)
        y = pd.concat(augmented_y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Base Models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
    svm = SVC(probability=True, random_state=42)
    
    # Voting Classifier
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gbm', gbm), ('svm', svm)],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    
    # Evaluation
    preds = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    print(f"Ensemble Model Accuracy: {accuracy:.4f}")
    
    # Feature Importance (Proxy from RF)
    rf.fit(X_train, y_train)
    feature_importances = dict(zip(features, rf.feature_importances_))
    
    # Save model
    models_dir = os.path.join(settings.BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'crop_health_ensemble.joblib')
    joblib.dump(ensemble, model_path)
    
    # Save metrics
    metrics_path = os.path.join(settings.BASE_DIR, 'output', 'model_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(classification_report(y_test, preds))
    
    return ensemble, feature_importances

@time_it
def train_yield_regressor(df):
    """
    Trains a Random Forest Regressor to predict crop yield.
    """
    print("Training Yield Predictor...")
    
    features = ['mean_red', 'mean_nir', 'NDVI', 'temperature', 'soil_moisture', 'moisture_temp_interaction']
    target = 'yield_tons_ha'
    
    X = df[features]
    y = df[target]
    
    # Augment if small dataset
    if len(df) < 50:
        augmented_X = [X]
        augmented_y = [y]
        for _ in range(20):
            noise = np.random.normal(0, 0.05, X.shape)
            augmented_X.append(X + noise)
            augmented_y.append(y + np.random.normal(0, 0.1, y.shape))
        X = pd.concat(augmented_X)
        y = pd.concat(augmented_y)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    preds = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"Yield Model - RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    model_path = os.path.join(settings.BASE_DIR, 'models', 'yield_regressor.joblib')
    joblib.dump(rf, model_path)
    
    return rf
