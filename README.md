# ML Assignment 2 - Breast Cancer Classification

#### a. Problem Statement

The goal is to classify breast cancer tumors as either **Malignant** or **Benign** based on cell measurements derived from fine needle aspirate (FNA) of a breast mass.

#### b. Dataset Description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** Kaggle - https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **Features:** 30 numeric features computed from digitized images of FNA
- **Instances:** 569 samples
- **Target:** Binary classification (Malignant vs Benign)

#### c. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest (Ensemble) | 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| XGBoost (Ensemble) | 0.9649 | 0.9950 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

#### d. Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieves highest AUC (0.996) indicating excellent class separation. Strong balanced performance with 96.49% accuracy and good recall (0.9286). Works well as a baseline linear model for this dataset. |
| Decision Tree | Lowest performer with 92.11% accuracy and lowest AUC (0.9448). Prone to overfitting and struggles with recall (0.8333), missing more malignant cases. Not recommended for medical diagnosis. |
| kNN | Good overall performance with 95.61% accuracy. Distance-based approach works well with scaled features. Balanced precision (0.9744) and recall (0.9048) make it reliable for this task. |
| Naive Bayes | Tied for lowest accuracy (92.11%) but has good AUC (0.9891). Independence assumption may not hold for correlated features. Lower precision (0.9231) leads to more false positives. |
| Random Forest (Ensemble) | Best overall model with highest accuracy (97.37%), perfect precision (1.0), and best MCC (0.9442). Ensemble approach reduces overfitting and provides robust predictions. Recommended for deployment. |
| XGBoost (Ensemble) | Second-best performer with 96.49% accuracy and perfect precision (1.0). Gradient boosting effectively captures complex patterns. Slightly lower recall (0.9048) than Random Forest. |
