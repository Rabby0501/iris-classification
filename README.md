# Iris Flower Classification 🌸

A machine learning experiment to classify Iris flower species (Setosa, Versicolor, Virginica) using the classic Iris dataset. The project explores data preprocessing, hyperparameter tuning, model evaluation, and feature importance analysis with a Random Forest classifier.

## 📊 Dataset

The Iris dataset contains 150 samples of iris flowers with 4 numerical features:

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Target classes:
- 0: Setosa
- 1: Versicolor
- 2: Virginica

## 🚀 Project Workflow

1. **Data Preparation** – Load dataset using `sklearn.datasets.load_iris()` and display first few rows.
2. **Data Cleaning & Visualization** – Generate boxplots to check for outliers.
3. **Data Normalization** – Standardize features using `StandardScaler` (mean=0, variance=1).
4. **Hyperparameter Tuning** – Use `GridSearchCV` to find optimal `n_estimators` and `max_depth` for Random Forest.
5. **Model Training & Evaluation** – Train Random Forest with best parameters, compute accuracy, and plot confusion matrix.
6. **Feature Importance** – Visualize the contribution of each feature to the model.
7. **Cross-Validation** – Perform 5‑fold cross‑validation to ensure robustness.

## 📈 Results

- **Best Hyperparameters**: `{'max_depth': None, 'n_estimators': 100}`
- **Test Accuracy**: `1.0` (100%)
- **Average Cross‑Validation Score**: `0.9667`

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### Feature Importance

![Feature Importance](feature_importance.png)

### Boxplot of Features

![Boxplot](boxplot.png)

## 🛠️ Requirements

Install the required packages using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn
