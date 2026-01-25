"""
Breast Cancer Classification - Streamlit Web Application
This app demonstrates 6 ML classification models for breast cancer diagnosis.

Features:
- Dataset upload option (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix and classification report
- Download sample test data
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    [data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    [data-testid="stMetricLabel"] {
        color: #333;
    }
    [data-testid="stMetricValue"] {
        color: #FF4B4B;
    }
    [data-testid="stMetricDelta"] {
        background-color: #000;
    }
</style>
""", unsafe_allow_html=True)


# Model file paths
MODEL_FILES = {
    'Logistic Regression': 'model/logistic_regression_model.pkl',
    'Decision Tree': 'model/decision_tree_model.pkl',
    'kNN': 'model/knn_model.pkl',
    'Naive Bayes': 'model/naive_bayes_model.pkl',
    'Random Forest': 'model/random_forest_model.pkl',
    'XGBoost': 'model/xgboost_model.pkl'
}


@st.cache_resource
def load_model(model_name):
    """Load a trained model from disk."""
    try:
        model = joblib.load(MODEL_FILES[model_name])
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_FILES[model_name]}. Please run train_models.py first.")
        return None


@st.cache_resource
def load_scaler():
    """Load the trained scaler."""
    try:
        scaler = joblib.load('model/scaler.pkl')
        return scaler
    except FileNotFoundError:
        st.error("Scaler file not found. Please run train_models.py first.")
        return None


@st.cache_resource
def load_feature_cols():
    """Load the feature columns."""
    try:
        feature_cols = joblib.load('model/feature_cols.pkl')
        return feature_cols
    except FileNotFoundError:
        st.error("Feature columns file not found. Please run train_models.py first.")
        return None


@st.cache_data
def load_model_results():
    """Load the model comparison results."""
    try:
        results_df = pd.read_csv('model/model_results.csv', index_col=0)
        return results_df
    except FileNotFoundError:
        return None


@st.cache_data
def load_sample_test_data():
    """Load sample test data for download."""
    try:
        test_data = pd.read_csv('data/test_data.csv')
        return test_data
    except FileNotFoundError:
        return None


def evaluate_predictions(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['AUC'] = 0.0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign (B)', 'Malignant (M)'],
                yticklabels=['Benign (B)', 'Malignant (M)'])
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🎗️ Breast Cancer Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML Assignment 2 - Binary Classification using Multiple Models</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Model selection dropdown
    st.sidebar.subheader("🤖 Select Model")
    selected_model = st.sidebar.selectbox(
        "Choose a classification model:",
        list(MODEL_FILES.keys()),
        help="Select one of the 6 trained classification models"
    )
    
    st.sidebar.markdown("---")
    
    # Download sample test data
    st.sidebar.subheader("📥 Sample Test Data")
    sample_data = load_sample_test_data()
    if sample_data is not None:
        csv = sample_data.to_csv(index=False)
        st.sidebar.download_button(
            label="⬇️ Download Test CSV",
            data=csv,
            file_name="sample_test_data.csv",
            mime="text/csv",
            help="Download sample test data to upload and test the models"
        )
        st.sidebar.info(f"Sample data contains {len(sample_data)} records")
    else:
        st.sidebar.warning("Sample test data not available. Run train_models.py first.")
    
    st.sidebar.markdown("---")
    
    # About section
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.markdown("""
    **Dataset:** Breast Cancer Wisconsin (Diagnostic)
    
    **Task:** Binary Classification
    - Benign (B) = 0
    - Malignant (M) = 1
    
    **Features:** 30 numeric features computed from digitized images of breast mass FNA
    
    **Models Implemented:**
    1. Logistic Regression
    2. Decision Tree
    3. K-Nearest Neighbors
    4. Naive Bayes (Gaussian)
    5. Random Forest (Ensemble)
    6. XGBoost (Ensemble)
    """)
    
    # Main content
    tab1, tab2 = st.tabs(["🔍 Make Predictions", "📊 Model Comparison"])
    
    # Tab 1: Make Predictions
    with tab1:
        st.header("🔍 Make Predictions")
        st.markdown(f"**Selected Model:** {selected_model}")
        
        # File upload
        st.subheader("📂 Upload Test Data (CSV)")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with test data",
            type=['csv'],
            help="Upload a CSV file containing the features for prediction. The file should have the same columns as the training data."
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded data
                uploaded_data = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {uploaded_data.shape}")
                
                # Show preview
                with st.expander("📋 Preview Uploaded Data", expanded=True):
                    st.dataframe(uploaded_data.head(10), use_container_width=True)
                
                # Load model and scaler
                model = load_model(selected_model)
                scaler = load_scaler()
                feature_cols = load_feature_cols()
                
                if model is not None and scaler is not None and feature_cols is not None:
                    # Check for target column
                    has_target = 'diagnosis' in uploaded_data.columns
                    
                    # Prepare features
                    if has_target:
                        y_true_labels = uploaded_data['diagnosis']
                        y_true = y_true_labels.map({'B': 0, 'M': 1})
                        X = uploaded_data.drop(columns=['diagnosis'])
                    else:
                        X = uploaded_data.copy()
                        y_true = None
                    
                    # Check for required columns
                    missing_cols = set(feature_cols) - set(X.columns)
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {missing_cols}")
                    else:
                        # Select and order columns
                        X = X[feature_cols]
                        
                        # Scale features
                        X_scaled = scaler.transform(X)
                        
                        # Make predictions
                        if st.button("🚀 Run Predictions", type="primary"):
                            with st.spinner("Making predictions..."):
                                y_pred = model.predict(X_scaled)
                                y_pred_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                            
                            # Show predictions
                            st.subheader("🎯 Prediction Results")
                            
                            # Create results dataframe
                            results = pd.DataFrame({
                                'Prediction': ['Malignant (M)' if p == 1 else 'Benign (B)' for p in y_pred],
                                'Confidence': [f"{max(model.predict_proba(X_scaled)[i])*100:.1f}%" for i in range(len(y_pred))] if hasattr(model, 'predict_proba') else ['N/A'] * len(y_pred)
                            })
                            
                            if has_target:
                                results['Actual'] = ['Malignant (M)' if t == 1 else 'Benign (B)' for t in y_true]
                                results['Correct'] = ['✅' if p == t else '❌' for p, t in zip(y_pred, y_true)]
                            
                            st.dataframe(results, use_container_width=True)
                            
                            # Summary statistics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Total Predictions", len(y_pred))
                                st.metric("Predicted Malignant", sum(y_pred))
                                st.metric("Predicted Benign", len(y_pred) - sum(y_pred))
                            
                            with col2:
                                if has_target:
                                    correct = sum(y_pred == y_true)
                                    st.metric("Correct Predictions", f"{correct}/{len(y_pred)}")
                                    st.metric("Accuracy", f"{correct/len(y_pred)*100:.2f}%")
                            
                            # If we have true labels, show metrics
                            if has_target and y_true is not None:
                                st.markdown("---")
                                st.subheader("📊 Evaluation Metrics")
                                
                                metrics = evaluate_predictions(y_true, y_pred, y_pred_proba)
                                
                                # Display metrics
                                cols = st.columns(6)
                                metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
                                for i, (name, col) in enumerate(zip(metric_names, cols)):
                                    if name in metrics:
                                        col.metric(name, f"{metrics[name]:.4f}")
                                
                                # Confusion Matrix
                                st.markdown("---")
                                st.subheader("📉 Confusion Matrix")
                                fig = plot_confusion_matrix(y_true, y_pred, selected_model)
                                st.pyplot(fig)
                                
                                # Classification Report
                                st.markdown("---")
                                st.subheader("📋 Classification Report")
                                report = classification_report(y_true, y_pred, target_names=['Benign (B)', 'Malignant (M)'])
                                st.code(report)
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        else:
            st.info("👆 Please upload a CSV file to make predictions.")
            st.markdown("""
            **Expected CSV format:**
            - The CSV should contain the 30 feature columns used for training
            - Optionally include a 'diagnosis' column (B/M) to evaluate predictions
            - Download the sample test data from the sidebar to see the expected format
            """)
    
    # Tab 2: Model Comparison
    with tab2:
        st.header("📊 Model Comparison Table")
        st.markdown("Comparison of all 6 classification models on the test dataset.")
        
        results_df = load_model_results()
        if results_df is not None:
            # Format and display results
            styled_df = results_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=0)
            st.dataframe(styled_df, use_container_width=True)
            
            # Best model highlight
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_acc_model = results_df['Accuracy'].idxmax()
                best_acc = results_df['Accuracy'].max()
                st.metric("🏆 Best Accuracy", f"{best_acc:.4f}", best_acc_model)
            
            with col2:
                best_auc_model = results_df['AUC'].idxmax()
                best_auc = results_df['AUC'].max()
                st.metric("🏆 Best AUC", f"{best_auc:.4f}", best_auc_model)
            
            with col3:
                best_f1_model = results_df['F1'].idxmax()
                best_f1 = results_df['F1'].max()
                st.metric("🏆 Best F1 Score", f"{best_f1:.4f}", best_f1_model)
            
            # Bar chart comparison
            st.markdown("---")
            st.subheader("📉 Visual Comparison")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            results_df.plot(kind='bar', ax=ax, colormap='Set2')
            ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.warning("⚠️ Model results not found. Please run `python train_models.py` first to train the models.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎓 ML Assignment 2 - BITS AIML/DSE M.Tech Program</p>
        <p>Breast Cancer Wisconsin (Diagnostic) Dataset | Binary Classification</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
