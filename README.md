# 💳 Credit Card Fraud Detection — A Binary Classification Problem

## Project Overview ✨

This project focuses on detecting fraudulent credit card transactions, a critical task in the financial industry. The primary challenge is extreme class imbalance, where fraudulent transactions make up a tiny fraction of the total. This notebook explores several techniques for handling that imbalance, including resampling strategies and cost-sensitive learning, before selecting a final model: XGBoost tuned with Bayesian optimization using Optuna, followed by decision-threshold selection on the validation set.

**Goal:** Minimize missed fraud cases (false negatives) while maintaining high precision. Missed fraud can cause direct financial loss, but excessive false alarms can overwhelm a fraud review team and erode trust in the system.

## Data Source 📊

The [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, provided by Worldline and ULB (Université Libre de Bruxelles). It contains transactions made by European cardholders over two days in September 2013.

**Features:** `Time` (seconds since the first transaction, later dropped), `V1` to `V28` (PCA-transformed features, anonymized for confidentiality), `Amount`, and `Class` (target: `0` = legitimate, `1` = fraud).

## Methodology 🛠️

### Data Cleaning & Preprocessing

- Removed 1,081 duplicate transaction entries.
- Confirmed there were no missing values.
- Standardized column names to lowercase.
- Dropped the `Time` column, since it was not used as a predictor in this analysis.
- Applied `StandardScaler` to the features, fitting the scaler on the training data only.

### Rigorous Evaluation Methodology

To reduce overly optimistic performance estimates, this project uses a three-way train/validation/test split:

- **Training set:** Used to fit the models.
- **Validation set:** Used to compare techniques, tune XGBoost hyperparameters with Optuna, and select the decision threshold.
- **Test set:** Reserved for the final evaluation after model selection.

This separation helps prevent the test set from influencing modeling decisions and provides a more reliable estimate of how the selected model generalizes to unseen data.

### Class Imbalance: The Core Challenge

After duplicate removal, the dataset contains:

- Legitimate transactions: **283,253 (99.83%)**
- Fraudulent transactions: **473 (0.17%)**

A model that predicts every transaction as legitimate would achieve approximately 99.8% accuracy while detecting zero fraud cases. This makes accuracy alone misleading. Precision and recall for the minority (fraud) class are much more informative.

### Addressing Class Imbalance: Results (Validation Set)

| Technique | Precision | Recall | F1 |
|---|---:|---:|---:|
| Baseline (Logistic Regression) | 91.2% | 68.4% | 0.782 |
| RandomOverSampler | 5.3% | 89.5% | 0.100 |
| SMOTE | 4.9% | 88.2% | 0.093 |
| ADASYN | 1.7% | 90.8% | 0.033 |
| RandomUnderSampler | 2.7% | 90.8% | 0.052 |
| EditedNearestNeighbours | 93.0% | 69.7% | 0.797 |
| Balanced Class Weights | 5.4% | 89.5% | 0.101 |
| Custom Class Weights (1:1000) | 3.5% | 90.8% | 0.067 |
| XGBoost | 94.2% | 85.5% | 0.897 |
| XGBoost + Optuna (threshold 0.5) | 98.5% | 85.5% | 0.915 |

Most resampling and class-weighting techniques increased recall above 88%, but precision fell into single digits, reaching as low as 1.7% with ADASYN. Although these approaches detected more fraud cases, they also generated many false alarms, making them less practical for a fraud review team.

EditedNearestNeighbours (ENN) was a notable exception. It improved on the baseline F1 score by removing selected observations near the decision boundary rather than simply duplicating minority-class records or removing majority-class records.

The strongest overall results came from XGBoost with `scale_pos_weight`, which accounted for class imbalance without resampling the training data. Optuna hyperparameter tuning improved precision further while maintaining recall at approximately 85.5% at the default 0.5 threshold.

### Threshold Optimization

Rather than relying on the default probability threshold of 0.5, I used the precision-recall curve on the validation set to select a threshold that maximized the F1 score.

The selected threshold was approximately **0.9721**. At this threshold, validation precision was approximately 98.5% and recall was approximately 85.5%.

Raising the threshold made the model more selective about flagging transactions as fraudulent. This improved precision but involved a trade-off: a higher threshold can also cause more actual fraud cases to be missed.

The threshold was selected using validation data, leaving the test set separate for the final evaluation.

## Final Result: Held-Out Test Set 📈

| Metric | Score |
|---|---:|
| Precision | 98.6% |
| Recall | 72.6% |
| F1 score | 0.836 |

On the held-out test set, the final model achieved 98.6% precision and 72.6% recall.

This means that approximately 98.6% of transactions flagged as fraudulent were actually fraudulent, while the model detected about 72.6% of all fraudulent transactions in the test set.

The results demonstrate a clear trade-off. The model keeps false positives low, but it still misses approximately 27% of actual fraud cases. That limitation matters because missed fraud can lead to financial losses.

In fraud detection, false positives also carry operational costs. They consume review resources and can erode trust in fraud alerts. Prioritizing precision makes the model's alerts more reliable, but this comes at the expense of detecting every fraudulent transaction.

The final threshold therefore represents a precision-oriented operating point rather than a universally optimal solution. In a real deployment, the preferred threshold would depend on the relative costs of missed fraud, false alarms, and manual review.

## Key Takeaways

- **Accuracy can be misleading:** With only 0.17% fraudulent transactions, a model can achieve very high accuracy while detecting no fraud at all.
- **Precision and recall must be considered together:** Resampling and class weighting often increased recall but substantially reduced precision in the approaches tested.
- **Algorithm choice mattered:** XGBoost with built-in class-imbalance handling outperformed the resampling approaches evaluated.
- **Hyperparameter tuning improved results:** Optuna improved the tuned XGBoost model's precision while maintaining strong recall at the default threshold.
- **Threshold selection changes the trade-off:** Selecting a higher threshold improved precision but reduced the proportion of fraud cases detected on the held-out test set.
- **Evaluation separation matters:** Hyperparameter tuning and threshold selection used validation data, while the test set was reserved for the final performance evaluation.

Overall, this project demonstrates an end-to-end binary classification workflow, from data cleaning and class-imbalance analysis to model comparison, Bayesian hyperparameter optimization, threshold selection, and held-out evaluation. It also highlights an important practical lesson: a useful fraud detection model must balance detection performance with the operational consequences of its predictions.
