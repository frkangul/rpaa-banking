# Improving Loan Portfolio Profitability for a Bank

## Getting Started

### Prerequisites
- Python 3.11.7
- Libraries: pandas, matplotlib, seaborn, scikit-learn, optuna, xgboost, graphviz, shap, hydra-core

### Installation
1. Clone the repository to your local machine.
2. Install the required dependencies using Poetry:
```bash
git clone https://github.com/frkangul/rpaa-banking.git
cd rpaa-banking
pyenv install 3.11.7
pyenv local 3.11.7
python -m venv .venv
source .venv/bin/activate
poetry install
```

## Usage

Run the main training script after setting configurations from the `config` dir:
```bash
python train.py
```

This will launch a training pipeli and create result directory with the given name via configuration files.

## Given Facts by the RPAA Team:

### Task: 
- Improve the profitability of the loan portfolio

### Profit Info:
- A personal loan is profitable if the customer repays the installments. If a customer defaults and the bank is not able to recover the lended money, it causes a loss to the bank.

### Delinquencies Info: 
- Loan delinquencies are defined as the number of installments not paid after the due date. When the delinquency exceeds 3 months (=3 installments, 3 payments missed), the loan is defined as defaulted.

### Assumptions
- The loan amount can be considered lost when a default occurs.
- For portfolio management only the first 12 installments can be considered. Delinquencies on later installments are not provided, and hence can be ignored.

## My Strategy

To solve the recruiting case for the RPAA team, I will outline a plan to build a predictive model that aims to maximize the future profits of the bank's personal loan portfolio. The model will predict the likelihood of a default loan based on the provided datasets. Here's how I would approach the task:


### Step 1: Problem Definition

The primary goal is to predict the probability of a loan becoming default within the first 12 installments. This will help the bank to make informed decisions on whether to approve a loan application, adjust the loan terms, or take preemptive actions to mitigate potential losses.

### Step 2: Data Exploration and Preprocessing

- **loan_data.csv**: Analyze the distribution of loan reasons, terms, requested amounts, and customer account numbers. Check for missing values and outliers.

- **customer_data.csv**: Examine customer demographics and their relationship with the bank. Convert `birth_date` and `joined_BANK_date` to age and length of relationship with the bank.

- **customer_financials.csv**: Calculate average salary, current account balance, saving account balance, and credit card balance for each customer over the provided period.

- **loan_delinquencies.csv**: Create a target variable indicating whether a loan has become default (1) or not (0) within the first 12 installments.

### Step 3: Feature Engineering

- Create new features such as debt-to-income ratio, average monthly balance, and credit utilization from the financials data.

- Encode categorical variables like `loan_reason`, `gender`, `religion`, and `employment` using one-hot encoding or label encoding.

- Consider interaction terms if they make economic sense (e.g., loan amount vs. salary).

### Step 4: Model Selection

Given the binary nature of the target variable (default or not), I would consider the following algorithms:

- Logistic Regression: As a baseline model due to its simplicity and interpretability.

- Gradient Boosting Machines (e.g., XGBoost (compatible with M1 series MACs)): For their performance in classification tasks.

### Step 5: Model Training and Validation

- Split the data into training and validation sets.

- Standardize numerical features to have a mean of zero and a standard deviation of one.

- Train the models using the training set and perform hyperparameter tuning using cross-validation.

- Evaluate the models on the validation set using appropriate metrics (e.g., Recall and AUC precision-recall).

### Step 6: Model Interpretation and Selection

- Analyze the feature importances to understand the drivers of loan default.

- Select the model with the best balance between performance and interpretability.

### Step 7: Presentation and Reporting
- Prepare a presentation summarizing the approach, model performance, and key findings.
- Discuss potential next steps and improvements if more time were available, such as:
  - Incorporating additional data sources (e.g., credit scores, macroeconomic indicators).
  - Exploring more complex models or ensemble methods.
  - Conducting a cost-benefit analysis to optimize the threshold for classifying a loan as high risk.

## Data Description

> Disclaimer: the data provided for the purpose of this exercise is in no way referred to actual bank clients, but it is rather based on simulations.

The data should be in `data/`.

### loan_data.csv

Contains all the information about the all the loans disbursed in the years 2016, 2017 and 2018.

- `loan_id`: unique identifier of the loan
- `cust_id`: unique identifier of the customer that requested the loan
- `date`: month and year when the loan has been disbursed. Example: 2017-09-01 means that the loan has been requested and disbursed in the month of September 2017
- `loan_reason`: what is the loan meant for. Categories go to:
    - Car: Financing of a new vehicle
    - Housing: Related to any house related expense (renovation, new furniture…)
    - Financial: repayment of some other debt
    - Personal: Personal Reasons
- `loan_term`: number of installments (in months)
- `requested_amount`: total lended amount in Euros
- `installment`: monthly installment amount, in Euros
- `number_accounts`: estimated number of non debt related products that the customer has at the time of the loan application. The considered products include current accounts, saving accounts and investment accounts.
- `number_loan_accounts`: estimated number of debt-related products that the customer has at the time of the loan application. The considered products include other personal loans and mortgages.

### customer_data.csv

Information about the customer, extracted from the database in January 2020:

- `cust_id`: unique identifier of the customer that requested the loan
- `gender`: gender of the customer (M for Male, F for Female)
- `religion`: if known, religion of the customer
- `employment`: employment sector for the customer
- `postal_code`: first two digits of the postal code, where the second is always set to 0
- `birth_date`: month and year of birth of the customer
- `joined_BANK_date`: month and year when the customer became a BANK client
- `number_client_calls_to_BANK`: total number of times the customer has reached BANK by phone
- `number_client_calls_from_BANK`: number of times the customer has been contacted by phone by BANK

### customer_financials.csv

Monthly information about product balances and detected salaries of the customer, for the years 2016, 2017 and 2018

- `cust_id`: unique identifier of the customer that requested the loan
- `date`: date when the date has been extracted.
- `salary`: detected salary relative to this date: If the date is 01-02-2017 (1st February 2017), this refers to the detected salary of the previous month (January 2017)
- `current_account_balance`: balance of all the current accounts on the extraction day at midnight.
Example: If the date is 01-02-2017 (1st February 2017), this refers to the balance on all the current accounts on the 1st of February 2017, at 00:01 AM CET
- `saving_acc_balance`: balance of the saving accounts on the extraction day at midnight.
Example:  If the date is 01-02-2017 (1st February 2017), this refers to the balance on all the current accounts on the 1st of February 2017, at 00:01 AM CET
- `credit_card_balance`: due balance of the credit card on the extraction day at midnight. 
Example: If the date is 01-02-2017 (1st February 2017), this refers to the balance on all the current accounts on the 1st of February 2017, at 00:01 AM CET.

### loan_delinquencies.csv

Contains the information of the latest deliquency on the installment. This data is provided only for the first 12 installments, in accordance with the banking rules.

- `loan_id`: unique identifier of the loan
- `start_date`: month and year when the delinquent installment was due
- `end_date`: month and year then the delinquent installment has been repaid. In case that the delinquent installment has not been repaid within 3 months, the date is set to the 4th month, no matter when and if the customer has repaid the delinquency.
