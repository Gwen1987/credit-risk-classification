# Credit Risk Analysis Report

## Overview of the Analysis

The goal of this analysis was to build and evaluate a machine learning model to predict credit risk based on a borrower's financial characteristics. Specifically, we aimed to classify loan applicants as either <b>high-risk (1) </b> - likely to default, or <b> healthy (0) </b> - likely to repay.  

We used the <i> lending_data.csv </i> dataset, which contained the following features:
* Loan Size
* Interest rate
* Borrower's income
* Debt-to-income ratio
* Number of accounts
* Total debt

The <b> target variable </b> was <i> loan_status </i>, where:
* 0 represents a <b> healthy loan </b>
* 1 represents a <b> high-risk loan </b>

Initial analysis using <i><b> `value_counts()` </i></b> showed a class imbalance, with healthy loans(0) being much more common than high-risk loans(1)


## Machine Learning Process

We followed a typical supervised learning workflow:
1. <b> Data Preprocessing: </b> Split the dataset into features`(X)` and target `(y)`
2. <b> Train-Test split: </b> Split the data into training and testing subsets using `train_test_split`
3. <b> Model Training: </b> Applied a  `LogisticsRegression` classifier using scikit-learn
4. <b> Model Evaluation: </b>Used a classification report to assess precision, recall, and F1-score and a `Confusion Matrix` to assess accuracy.





## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* <b> Machine Learning Model 1: </b>
    * <b> Accuracy: </b> 99%
    * <b> Precision: </b>
        * Healthy loan (0): 100% 
        * High-risk loan (1): 84%
    * <b> Recall: </b>
        * Healthy loan (0): 99%
        * High-risk loan (1): 94%
    * <b> F1-Score: </b>
        * Healthy loan (0): 100%
        * High-risk loan (1): 89%
    * <b> Support: </b>
        * Healthy loans: 18,765
        * High-risk loans: 619


## Summary

The Logistic Regression model achieved <b> excellent overall performance </b>, particularly given the class imbalance. It maintained a <b> high recall (0.94) </b> and <b> F1-score (0.89) </b> for the <b> high-risk loan class (1), </b> indicating that it correctly identified most high-risk cases without severely compromising precision.

## Recommendation:
* This model is <b> suitable for production </b> or decision support systems where <b> catching high-risk loans is important </b>, such as in loan underwriting or credit checks.

* Although logistic regression is a simple algorithm, it performed very well here. If further improvement is needed, you may still consider:

    * Trying more complex models (e.g., Random Forest, XGBoost)
    * Performing hyperparameter tuning
    * Monitoring for drift or performance drops on new data

Given the balance of precision and recall for both classes, <b> this Logistic Regression model is recommended </b> for detecting credit risk in this dataset.


