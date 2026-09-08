# 💳 Credit Card Fraud Detection — A Binary Classification Problem

## Project Overview ✨

This project focuses on detecting fraudulent credit card transactions, a critical task in the financial industry. The primary challenge in this domain is extreme class imbalance, where fraudulent transactions make up a tiny fraction of the total. This notebook explores several techniques for handling that imbalance, including multiple resampling strategies and cost sensitive learning, before landing on a final model: XGBoost tuned with Bayesian optimization (Optuna) and a threshold selected from the validation set.

**Goal:** build a model that minimizes missed fraud cases (false negatives) as much as possible without letting precision collapse. Missed fraud causes direct financial loss, but too many false alarms overwhelm a fraud review team and erode trust in the system.

## Data Source 📊

The [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, provided by Worldline and ULB (Université Libre de Bruxelles). Contains transactions made by European cardholders over two days in September 2013.

**Features:** `Time` (seconds since the first transaction, later dropped), `V1 to V28` (PCA transformed features, anonymized for confidentiality), `Amount`, and `Class` (target: 0 = legitimate, 1 = fraud).

## Methodology 🛠️

### Data Cleaning & Preprocessing
- Removed 1,081 duplicate transaction entries
- Confirmed there were no missing values
- Standardized column names to lowercase
- Dropped the `Time` column, since it isn't directly indicative of fraud
- Applied `StandardScaler` to the features, fit only on training data

### Rigorous Evaluation Methodology

To keep the reported results honest rather than optimistic, this project uses a three way train / validation / test split:
- **Training set**, used to fit every model
- **Validation set**, used for all technique comparisons, hyperparameter tuning (Optuna), and threshold selection
- **Test set**, touched exactly once at the very end, to report the final performance of the model that had already been fully selected

This avoids a common pitfall in imbalanced classification work, where tuning decisions get made against the same data used to report final performance, quietly inflating the results.

### Class Imbalance: The Core Challenge

- Legitimate transactions: 283,253 (99.83%)
- Fraudulent transactions: 473 (0.17%)

A model that predicts "legitimate" for every transaction would score 99.8% accuracy while catching zero fraud. That makes standard accuracy meaningless here, and precision and recall the metrics that actually matter.

### Addressing Class Imbalance: Results (validation set)

| Technique | Precision | Recall | F1 |
|---|---|---|---|
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

Every resampling and class weighting technique followed the same pattern: recall climbed above 88%, but precision collapsed into single digits, as low as 1.7% for ADASYN. That means the model would flag overwhelming numbers of legitimate transactions as fraud. EditedNearestNeighbours was the one method that meaningfully beat the baseline, since it cleans noisy data near the decision boundary instead of blindly duplicating or removing records. The real improvement came from changing the algorithm rather than the data. XGBoost with `scale_pos_weight` outperformed every resampling technique tried, and Bayesian tuning with Optuna improved it further still.

### Threshold Optimization

Instead of using the default 0.5 cutoff, the decision threshold was tuned using the precision recall curve on the validation set, to find the point that best reflects the real cost of missed fraud versus false alarms.

## Final Result: Held Out Test Set (evaluated once) 📈

| Metric | Score |
|---|---|
| Precision | 98.6% |
| Recall | 72.6% |
| F1 score | 0.836 |

When the model flags a transaction as fraud, it's correct 98.6% of the time, which nearly eliminates false alarms, while still catching about 3 out of every 4 actual fraud cases.

In fraud detection, false positives carry a real operational cost. They erode trust in the system and overwhelm the people who have to act on every alert. Prioritizing precision here means the alerts this model produces are ones a fraud team can actually act on with confidence, rather than a flood of false alarms that eventually gets ignored. That was a deliberate business trade-off, not just a statistical one.

## Key Takeaways

- Naive accuracy is meaningless under 0.17% class imbalance. Precision and recall on the minority class are what actually matter.
- Resampling and class weighting techniques can boost recall dramatically, but they often destroy precision in the process. The two need to be evaluated together, not separately.
- Algorithm choice mattered more than data resampling. XGBoost with built in imbalance handling outperformed every resampling approach tested.
- All tuning decisions were made on a validation set. The test set was used exactly once, so the final reported performance reflects genuine generalization rather than accidental overfitting to the evaluation data.
