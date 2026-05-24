# Credit Risk Scoring & Default Risk Segmentation

## Project Overview

This project develops a **business-oriented credit risk scoring pipeline** for predicting whether a credit card client is likely to default in the next month.

The goal is not only to build a classification model, but also to translate model probabilities into practical business outputs such as:

- credit default risk scores,
- decision thresholds,
- customer risk bands,
- risk deciles,
- model explainability,
- and business recommendations for credit analyst review.

This makes the project closer to a real **banking / credit risk analytics** use case rather than a simple machine learning benchmark.

---

## Business Problem

Financial institutions need to identify clients with an increased probability of default in order to reduce credit losses and improve risk-based decision making.

In this project, a model is trained to estimate the probability that a client will default on their credit card payment next month.

Because the cost of errors is asymmetric, the project focuses not only on accuracy, but also on:

- **Recall** — how many actual defaulting clients are detected,
- **Precision** — how many flagged clients are truly risky,
- **F1-score** — balance between precision and recall,
- **ROC-AUC** — ranking ability of the model,
- **Threshold tuning** — adapting the decision rule to business needs.

In credit risk, a false negative can be especially costly because it means that a risky client is incorrectly treated as safe.

---

## Dataset

The project uses the public **Default of Credit Card Clients** dataset.

Dataset source: [Default of Credit Card Clients Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

The dataset contains information about credit card clients, including:

- demographic variables,
- credit limit,
- repayment history,
- bill statement amounts,
- previous payment amounts,
- and default status in the next month.

Target variable:

```text
default.payment.next.month
```

Target interpretation:

```text
0 = no default
1 = default
```


## Project Structure

```text
credit-risk-scoring/
│
│   
│
├── notebooks/
│   └── credit_card_default_advanced_final.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Project Review

To review the project, open the notebook:

```text
notebooks/credit_card_default_advanced_final.ipynb
```

The notebook contains the full machine learning workflow, including:

- exploratory data analysis,
- data cleaning,
- feature engineering,
- model benchmarking,
- class imbalance handling,
- multicollinearity analysis,
- PCA dimensionality reduction,
- Boruta feature selection,
- threshold tuning,
- SHAP explainability,
- risk segmentation.

The project is intended mainly as a portfolio case study in credit risk analytics, banking analytics and machine learning.

---

## Methods

The project includes the following steps:

### 1. Exploratory Data Analysis

The notebook starts with data inspection, target distribution analysis and selected visualizations for key variables such as credit limit, age, repayment status and bill/payment amounts.

Since the target is imbalanced, accuracy alone is not treated as the main model quality measure.

### 2. Data Cleaning

Selected categorical variables are cleaned by grouping rare or undefined categories into broader groups.

Examples:

- `EDUCATION`
- `MARRIAGE`

### 3. Feature Engineering

Additional credit risk features are created to better represent repayment behavior and credit exposure, including:

- total bill amount,
- average bill amount,
- maximum bill amount,
- total previous payment amount,
- average previous payment amount,
- credit utilization ratios,
- payment-to-bill ratio,
- maximum payment delay,
- average payment delay,
- number of months with delayed payments.

These features are intended to describe the client’s financial behavior more directly than raw variables alone.

### 4. Model Benchmarking

Several machine learning models are trained and compared:

- Logistic Regression,
- Random Forest,
- XGBoost,
- LightGBM,
- CatBoost.

The models are evaluated using:

- accuracy,
- precision,
- recall,
- F1-score,
- ROC-AUC.

### 5. Class Imbalance Handling

The project includes an experiment with **SMOTETomek**, which combines synthetic minority oversampling with cleaning of overlapping observations.

This method is treated as an experimental comparison rather than automatically assumed to improve the final model.

### 6. Multicollinearity Analysis with VIF

VIF analysis is used to inspect multicollinearity, especially for linear models.

High or infinite VIF values are expected in this project because several engineered features are derived from original bill and payment variables.

For example:

- `total_bill_amt` is derived from `BILL_AMT1`–`BILL_AMT6`,
- `avg_bill_amt` is derived from bill amount columns,
- utilization ratios are derived from bill amounts and credit limit.

This is mainly a concern for linear models. Tree-based models such as CatBoost, LightGBM and XGBoost are less sensitive to multicollinearity.

### 7. PCA Dimensionality Reduction

PCA is tested as a dimensionality reduction approach.

In the executed notebook, PCA reduced the feature space from:

```text
41 original / engineered features
```

to:

```text
18 principal components explaining 95% of variance
```

However, tree-based models performed better on the original engineered features, so PCA was treated as an experiment rather than the final approach.

### 8. Boruta Feature Selection

Boruta feature selection is used to identify the most relevant variables.

The selected features were mainly related to:

- repayment history,
- payment delay variables,
- bill amounts,
- previous payment amounts,
- credit utilization,
- payment-to-bill ratio.

This supports the business interpretation that repayment behavior is more informative than demographic information.

### 9. Threshold Tuning

The default probability threshold of `0.5` is not necessarily optimal in credit risk.

The notebook compares different thresholds and includes:

- F1-based threshold selection,
- cost-sensitive threshold analysis,
- comparison between more conservative and more balanced strategies.

This allows the model to be adapted depending on whether the business wants to:

- detect as many risky clients as possible,
- reduce false alarms,
- or balance both objectives.

### 10. SHAP Explainability

SHAP is used to explain which variables influence model predictions.

This is important because credit risk models should not be treated as black boxes. Model explainability helps understand whether predictions are driven by meaningful financial behavior.

### 11. Risk Segmentation

The final model probabilities are transformed into risk bands:

- Very low,
- Low,
- Medium,
- High,
- Very high.

This makes the output easier to interpret from a business perspective and can support credit analyst workflows.

---

## Final Model

The final selected model was:

```text
CatBoost + Boruta feature selection
```

Final test set results:

| Metric | Value |
|---|---:|
| Accuracy | 0.785 |
| Precision | 0.513 |
| Recall | 0.569 |
| F1-score | 0.539 |
| ROC-AUC | 0.780 |
| Decision threshold | 0.57 |

The final model was selected using a business-oriented scoring approach that considered not only ROC-AUC, but also F1-score and recall.

---

## Risk Segmentation Results

The final model produced meaningful customer risk bands:

| Risk band | Customers | Observed default rate | Average predicted probability |
|---|---:|---:|---:|
| Very low | 981 | 4.3% | 14.8% |
| Low | 1,724 | 10.6% | 27.9% |
| Medium | 1,450 | 18.5% | 41.0% |
| High | 927 | 28.8% | 59.6% |
| Very high | 918 | 61.8% | 83.8% |

The observed default rate increases clearly across the risk bands, which means that the model is useful for ranking customers by credit risk.

This is one of the most important business outcomes of the project.

---

## Business Decision Scenarios

The model can support different credit risk strategies depending on the selected threshold.

| Strategy | Example threshold | Business interpretation |
|---|---:|---|
| Conservative risk detection | ~0.37 | Detect more risky clients, accept more false positives |
| Balanced model | ~0.57 | Balance precision and recall |
| Strict flagging | >0.70 | Flag only the highest-risk clients |

In a real banking workflow, the model would not have to automatically reject customers.

It could instead create a **manual review queue** for credit analysts.

For example:

- top-risk clients can be reviewed manually,
- medium-risk clients can receive additional verification,
- low-risk clients can follow the standard process.

---

## Key Findings

1. The target variable is imbalanced, so accuracy alone is not sufficient.
2. Tree-based models performed better than a simple linear baseline.
3. CatBoost with Boruta-selected features was selected as the final model.
4. Repayment history and payment behavior were more informative than demographic variables.
5. PCA reduced dimensionality but did not improve the final tree-based model.
6. Risk segmentation showed a strong increase in observed default rate from very low to very high risk groups.
7. Threshold tuning is necessary because different thresholds represent different business strategies.

---

## Technologies Used

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- Boruta
- imbalanced-learn
- statsmodels
- Jupyter Lab

---

## GitHub Notes

The raw dataset and generated local artifacts are not included in this repository.

The repository is focused on:

- the final notebook,
- project methodology,
- model comparison,
- threshold tuning,
- SHAP explainability,
- risk segmentation,
- and business interpretation.

The dataset should be downloaded separately and placed locally in:

```text
data/raw/UCI_Credit_Card.csv
```

Some local files and folders are ignored by `.gitignore`, including:

```text
data/raw/
data/processed/
models/
reports/
catboost_info/
.ipynb_checkpoints/
```

---

## Limitations

This project is a portfolio prototype and should not be used as a production credit decision system without additional validation, monitoring, governance, fairness analysis and regulatory review.

In a real banking environment, additional steps would be required, such as:

- out-of-time validation,
- model monitoring,
- probability calibration,
- regulatory review,
- fairness and bias analysis,
- explainability review,
- data drift monitoring,
- production deployment controls.

---

## Summary

This project demonstrates an end-to-end credit risk machine learning workflow using Python.

It combines predictive modeling with explainability, threshold optimization and risk segmentation, making it suitable for a portfolio project in banking analytics, credit risk, data science or machine learning.
