## About This Project

This project is part of my Machine Learning learning journey.

Rather than only implementing algorithms, my goal is to understand the reasoning behind every engineering decision—from preprocessing and optimization to debugging and model evaluation.

I document these projects to track my learning, deepen my understanding, and share practical insights with others who are also building their foundations in Machine Learning.

---

# Employee Turnover Prediction using Logistic Regression

> A supervised Machine Learning project focused on building, evaluating, and improving Logistic Regression models through feature scaling, L1 (Lasso) and L2 (Ridge) regularization while understanding the engineering decisions behind each approach.

---

## Project Overview

Employee turnover is a significant challenge for organizations, leading to increased recruitment costs, knowledge loss, and project delays.

In this project, I developed a binary classification model capable of predicting whether an employee is likely to leave the company based on multiple workplace-related attributes.

Rather than stopping at building a baseline model, this project became an opportunity to explore **why** different preprocessing techniques and regularization methods influence Logistic Regression and how those decisions affect model performance.

---

## Objectives

- Build a baseline Logistic Regression model.
- Improve the model using feature scaling.
- Apply L1 (Lasso) Regularization.
- Apply L2 (Ridge) Regularization.
- Compare model performance using multiple evaluation metrics.
- Understand the reasoning behind regularization, optimization, solver selection, and cross-validation.

---

## Dataset

The dataset contains **900 employee records** with workplace-related information such as:

- Job Satisfaction
- Performance Rating
- Years at Company
- Work-Life Balance
- Monthly Income
- Age
- Education Level
- Annual Bonus
- Training Hours
- Department
- Employee Role

Additional engineered features include:

- Annual Bonus²
- Bonus × Training Hours Interaction

**Target Variable**

- Employee Turnover
    - 1 → Employee Left
    - 0 → Employee Stayed

---

## Machine Learning Pipeline

```
                               Employee Dataset
	                                  │
	                                  ▼
	                    Exploratory Data Analysis (EDA)
	                                  │
	                                  ▼
	                 Data Cleaning & Feature Engineering
	                                  │
	                                  ▼
	                      Train-Test Split (80/20)
	                     X_train, X_test, y_train, y_test
	                    ┌──────────────┴──────────────┐
	                    │                             │
	                    ▼                             ▼
	         Original Features                 StandardScaler
	                    │                             │
	                    ▼                             ▼
	     Baseline Logistic Regression      Scaled Features
	                    │                             │
	                    │          ┌──────────────────┼──────────────────┐
	                    │          │                  │                  │
	                    │          ▼                  ▼                  ▼
	                    │  Logistic Regression   Lasso Logistic CV   Ridge Logistic CV
	                    │      (Scaled)             (L1 + CV)            (L2 + CV)
	                    │          │                  │                  │
	                    └──────────┴──────────────────┴──────────────────┘
	                                       │
	                                       ▼
	                     Performance Comparison of All Models
	                                       │
	                                       ▼
	         Accuracy • Precision • Recall • F1-score • Confusion Matrix
	                                       │
	                                       ▼
                         Best Model Selection & Key Insights

```

---

## Models Implemented

### 1. Baseline Logistic Regression

A standard Logistic Regression model trained on the original dataset to establish a performance benchmark.

---

### 2. Scaled Logistic Regression

Applied **StandardScaler** before training.

This experiment helped me understand why optimization-based algorithms often converge better when numerical features have similar scales.

---

### 3. Lasso Logistic Regression (L1)

Implemented using

- LogisticRegressionCV
- penalty = "l1"
- solver = "saga"

Cross-validation was used to automatically determine the optimal inverse regularization strength (**C**).
<img width="1284" height="831" alt="image" src="https://github.com/user-attachments/assets/1938ec7a-3114-4169-b03a-3b28d6dae761" />


---

### 4. Ridge Logistic Regression (L2)

Implemented using

- LogisticRegressionCV
- penalty = "l2"

This model retains all features while shrinking coefficients to reduce model complexity.

---

## Performance Comparison

| Model | Accuracy |
|---------|---------:|
| Baseline | **85.93%** |
| Scaled Baseline | **86.30%** |
| Lasso (L1) | **85.56%** |
| Ridge (L2) | **85.93%** |

### Key Observation

Although all four models produced similar results, feature scaling provided a small improvement while Ridge regularization maintained comparable performance to the baseline. Lasso slightly reduced predictive performance, suggesting that removing certain features caused a minor loss of useful information rather than improving generalization.

---

## Challenges & Debugging

This project taught me as much through debugging as through implementation.

Some of the issues I encountered included:

### 1. Solver Compatibility

Initially attempted L1 regularization using the `lbfgs` solver.

Learned that:

- `lbfgs` only supports L2 regularization.
- L1 optimization requires solvers such as `saga` or `liblinear`.

---

### 2. Feature Name Warning

Encountered the warning:

```
X does not have valid feature names...
```

This happened because the baseline model was trained on a Pandas DataFrame while other models were trained on scaled NumPy arrays.

Maintaining separate scaled and unscaled datasets resolved the issue and reinforced the importance of keeping preprocessing pipelines consistent.

---

### 3. Model Evaluation

While comparing models, I initially obtained inconsistent classification reports.

The issue originated from evaluating different models using mismatched versions of the test data (scaled vs. unscaled).

Instead of duplicating evaluation code, I refactored the workflow to correctly select the appropriate test dataset for each model.

---

## Key Learnings

This project strengthened far more than my understanding of Logistic Regression.

It reinforced the importance of:

- treating preprocessing as part of the model rather than a separate step
- understanding optimization instead of viewing models as black boxes
- selecting evaluation metrics based on business requirements rather than accuracy alone
- debugging preprocessing pipelines carefully before trusting model performance
- asking *why* implementation decisions exist instead of simply applying them

Perhaps the biggest lesson was realizing that building a model is only one part of machine learning. Understanding the engineering decisions behind preprocessing, optimization, evaluation, and debugging is what transforms implementation into genuine learning.

---

## Future Improvements

Possible extensions include:

- ROC-AUC comparison
- Precision-Recall analysis
- Hyperparameter tuning using GridSearchCV
- Feature importance analysis
- Confusion matrix visualization
- Threshold tuning
- Deployment using Streamlit or Flask

---

## Concepts Explored Beyond Implementation

One of the goals of this project was not only to train models but also to understand the reasoning behind them.

Topics I explored during implementation include:

- Logistic Regression optimization
- Gradient Descent intuition
- Why Logistic Regression has no closed-form solution
- Why feature scaling affects optimization
- L1 vs L2 regularization
- Why L1 produces sparse models
- Why Ridge preserves all features
- Cross Validation
- Selecting the optimal C value
- Why LogisticRegressionCV uses **C** instead of λ
- Solver compatibility (`lbfgs`, `liblinear`, `saga`)
- Choosing appropriate evaluation metrics

---

## Repository Structure

```
├── notebooks/
│      project2_EmployeeTurnover.ipynb
├── README.md
└── requirements.txt
```

---
