# Credit Card Fraud Detection
Binary Classification

## Project Overview ✨
This project focuses on detecting fraudulent credit card transactions, a critical task in the financial industry. The primary challenge in this domain is the extreme class imbalance, where fraudulent transactions are a tiny fraction of the total. This notebook explores various techniques to handle this imbalance, including different resampling strategies and cost-sensitive learning, culminating in a robust XGBoost model optimized using Bayesian hyperparameter tuning with Optuna.

**Goal**: To build a machine learning model that accurately identifies fraudulent transactions while minimizing False Negatives (missed fraud cases), which can lead to significant financial losses.

## Data Source 📊
The dataset used for this project is the Credit Card Fraud Detection dataset from Kaggle, provided by Worldline and ULB (Université Libre de Bruxelles). It contains transactions made by European cardholders in September 2013 over two days.

### About the data 
The dataset features include:
* Time: Seconds elapsed between each transaction and the first transaction in the dataset.
* V1-V28: PCA-transformed features (due to confidentiality).
* Amount: Transaction amount.
* Class: The target variable, where 0 indicates a legitimate transaction and 1 indicates a fraudulent transaction.

## Methodology 🛠️
### Data Cleaning & Preprocessing
1. **Duplicate Removal**: Identified and removed 1081 duplicate transaction entries.
2. **Missing Values**: Confirmed the absence of any missing values.
3. **Column Renaming**: Converted all column names to lowercase for consistency.
4. **Feature Scaling**: Applied StandardScaler to the Amount and PCA-transformed V features to ensure uniform scale, which is crucial for many machine learning algorithms.
5. **Feature Dropping**: The Time column was dropped as it's often not directly indicative of fraud and the PCA features already capture temporal patterns indirectly.

###  Exploratory Data Analysis (EDA)
The EDA focused heavily on understanding the class distribution of the target variable (`Class`).
-   **Class Imbalance:** The dataset exhibits extreme imbalance:
    -   Legitimate Transactions (Class 0): 283,253
    -   Fraudulent Transactions (Class 1): 473 (approximately 0.17% of the total)
 
### Addressing Class Imbalance
The extreme class imbalance makes standard models perform poorly, often classifying everything as the majority class. I experimented with several techniques:

1.  **Resampling Methods (with Logistic Regression):**
    *   **`RandomOverSampler`:** Increased the number of minority class samples by randomly duplicating them.
        -   **Outcome:** Significantly reduced False Negatives (FN), but led to a very low Precision (0.06) and f1-score (0.11) due to a high number of False Positives (FP).
    *   **`SMOTE` (Synthetic Minority Over-sampling Technique):** Generated synthetic samples for the minority class based on its nearest neighbors.
        -   **Outcome:** Similar to `RandomOverSampler` with slightly more False Positives, resulting in a low f1-score (0.10).
    *   **`ADASYN` (Adaptive Synthetic Sampling):** Similar to SMOTE but focuses on generating samples for minority class instances that are harder to classify.
        -   **Outcome:** Achieved the highest Recall (0.92) but at the cost of extremely low Precision (0.02) and f1-score (0.03), indicating too many False Positives.
    *   **`RandomUnderSampler`:** Reduced the number of majority class samples by randomly removing them.
        -   **Outcome:** Resulted in a very low f1-score (0.09), as removing too much valuable majority class data led to poor overall learning.
    *   **`EditedNearestNeighbours` (ENN):** An undersampling method that cleans the decision boundary by removing noisy majority class samples.
        -   **Outcome:** Improved balance with Precision (0.86) and Recall (0.63) compared to random undersampling, but Recall was still deemed too low for fraud detection.

2.  **Cost-Sensitive Learning (with Logistic Regression):**
    *   **`class_weight='balanced'`:** Automatically adjusted weights inversely proportional to class frequencies.
        -   **Outcome:** Performance was similar to `RandomOverSampler`, indicating that simply balancing weights might not be sufficient for such extreme imbalance.
    *   **Custom `class_weight={0:1, 1:1000}`:** Manually assigned a much higher weight to the minority class.
        -   **Outcome:** Further decreased Precision (0.04) while Recall remained similar (0.87), making it less suitable due to high FPs.

### Model Selection & Hyperparameter Tuning

Given the limitations of various resampling and cost-sensitive methods with Logistic Regression, I moved to more powerful ensemble algorithms:

1.  **Support Vector Classification (SVC) & Random Forest:**
    *   **Outcome:** Initial attempts to use these models were commented out due to the large dataset size, which led to prohibitively long training times without extensive computational resources.

2.  **XGBoost (Extreme Gradient Boosting):**
    *   **Rationale:** XGBoost is known for its efficiency and performance on imbalanced datasets, offering parameters like `scale_pos_weight` to handle imbalance directly.
    *   **Initial XGBoost Model:**
        -   Used `scale_pos_weight` (ratio of negative to positive classes in the training data).
        -   **Outcome:** Achieved a strong balance: Precision 0.95, Recall 0.77, f1-score 0.85. This was the best performing model thus far.

3.  **Bayesian Optimization (Optuna) for XGBoost:**
    *   **Rationale:** To further optimize the XGBoost model, I used [Optuna](https://optuna.org/), a popular hyperparameter optimization framework, to find the best set of hyperparameters.
    *   **Objective:** Maximize the F1-score for the fraud class.
    *   **Hyperparameters Tuned:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `min_child_weight`.
    *   **Outcome:** Optuna identified a set of hyperparameters that improved the F1-score to **0.8837** at a default threshold of 0.5.

### Threshold Optimization
A critical step in fraud detection is adjusting the classification threshold, especially when Recall (detecting fraud) is prioritized over Precision (avoiding false alarms).

1.  **Precision-Recall Curve Analysis:** I plotted the precision and recall values across various probability thresholds to understand their trade-off.
2.  **Optimal Threshold Selection:** I identified an optimal threshold that maximizes the F1-score, providing the best balance between precision and recall for the fraud class.
    -   The `best_threshold` was determined to be **0.9444**.

## Results & Key Findings  📈

### Final Model Performance

The best model, **XGBoost with Optuna-tuned hyperparameters and an optimized threshold of 0.9444**.
