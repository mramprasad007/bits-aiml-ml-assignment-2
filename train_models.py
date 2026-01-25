"""
Train Models Script for Breast Cancer Classification
This script trains 6 classification models and saves them for use in the Streamlit app.

Models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import ML models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Load and prepare the breast cancer dataset.
def load_and_prepare_data(data_path='data/breast_cancer_data.csv'):
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Create a copy for cleaning
    df_cleaned = df.copy()
    
    # Handle missing values - impute numerical with median
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            median_value = df_cleaned[col].median()
            df_cleaned[col].fillna(median_value, inplace=True)
            print(f"Imputed '{col}' with median: {median_value}")
    
    # Impute categorical columns with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            mode_value = df_cleaned[col].mode()[0]
            df_cleaned[col].fillna(mode_value, inplace=True)
            print(f"Imputed '{col}' with mode: {mode_value}")
    
    # Remove duplicates
    duplicates = df_cleaned.duplicated().sum()
    if duplicates > 0:
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")
    
    return df_cleaned

# Prepare features and target variables
def prepare_features_target(df):
    # Encode target variable
    label_encoder = LabelEncoder()
    df['diagnosis_encoded'] = label_encoder.fit_transform(df['diagnosis'])
    
    print(f"\nTarget Variable Encoding:")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Mapping: B (Benign) = 0, M (Malignant) = 1")
    
    # Define columns to drop
    cols_to_drop = ['id', 'diagnosis', 'diagnosis_encoded']
    if 'Unnamed: 32' in df.columns:
        cols_to_drop.append('Unnamed: 32')
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in cols_to_drop]
    
    print(f"\nNumber of features: {len(feature_cols)}")
    
    # Create feature matrix and target vector
    X = df[feature_cols]
    y = df['diagnosis_encoded']
    
    return X, y, feature_cols, label_encoder

# Train and evaluate a model, returning all required metrics.
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\nTraining {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  AUC: {metrics['AUC']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1: {metrics['F1']:.4f}")
    print(f"  MCC: {metrics['MCC']:.4f}")
    
    return metrics, y_pred, model

# Main function to train and save all models.
def main():
    print("=" * 60)
    print("BREAST CANCER CLASSIFICATION - MODEL TRAINING")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Prepare features and target
    X, y, feature_cols, label_encoder = prepare_features_target(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    print(f"\nTrain-Test Split:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    print("\nFeature scaling complete!")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }
    
    # Train and evaluate all models
    results = []
    trained_models = {}
    predictions = {}
    
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    for model_name, model in models.items():
        metrics, y_pred, trained_model = evaluate_model(
            model, X_train_scaled, X_test_scaled, y_train, y_test, model_name
        )
        results.append(metrics)
        trained_models[model_name] = trained_model
        predictions[model_name] = y_pred
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('Model')
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(results_df.round(4).to_string())
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Save models
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    for model_name, model in trained_models.items():
        filename = f"model/{model_name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, filename)
        print(f"Saved: {filename}")
    
    # Save scaler
    joblib.dump(scaler, 'model/scaler.pkl')
    print("Saved: model/scaler.pkl")
    
    # Save feature columns
    joblib.dump(feature_cols, 'model/feature_cols.pkl')
    print("Saved: model/feature_cols.pkl")
    
    # Save label encoder
    joblib.dump(label_encoder, 'model/label_encoder.pkl')
    print("Saved: model/label_encoder.pkl")
    
    # Save results
    results_df.to_csv('model/model_results.csv')
    print("Saved: model/model_results.csv")
    
    # Save test data for Streamlit app testing
    test_data = X_test.copy()
    test_data['diagnosis'] = y_test.map({0: 'B', 1: 'M'})
    test_data.to_csv('data/test_data.csv', index=False)
    print("Saved: data/test_data.csv")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return results_df, trained_models

if __name__ == "__main__":
    results_df, trained_models = main()
