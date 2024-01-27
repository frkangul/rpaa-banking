# ING Machine Learning Case

This is the recruiting case for the RPAA team. Good luck & have fun!

## Use Case Description

The current lending manager responsible for personal loans at ING has engaged you to help improve the profitability of the loan portfolio.
His department has collected four datasets for you to use, detailed below.

### Assignment

Your goal as Data Scientist is to build a model that can help maximise the future profits. The modeling solution (definition of the problem, algorithms used, designs, etc.) is up to you.

> Should you have any questions concerning this case, we encourage you to reach out to Aydin Senaydin (aydin.senaydin@ing.com).

### Assessment

- Focus on solving a single modelling problem, otherwise the case would take too much time.
- Push your code to the master branch of your git repository.
- There will be a 30 minute presentation and ~30-45 minutes of Q&A. The storyline should be understandable for a 'lending manager' but deep dives on data science / machine learning are OK. There is no requirement for the form of your presentation (but PowerPoint is typical).
- Feel free reuse your own code from other projects, or reuse code from other sources.
- You can always spend more time building even better models. The case is intended to take ~8 hours. Let us know in the presentation which ideas you would pursue if you would have had more time to work on the project.
- Please submit code & presentation 24 hours before your interview so that we can review before the interview.

We will evaluate the quality of your code, your solution and how you present it to peers and business stakeholders.

## Quick primer on personal loans

Personal loans are financial products designed to provide financing to a customer. After a loan is given to a customer, they have to repay in monthly installments. 
A personal loan is profitable if the customer repays the installments.
If a customer defaults and the bank is not able to recover the lended money, it causes a loss to the bank (see details in the delinquencies section).

### Delinquencies

Loan delinquencies are defined as the number of installments not paid after the due date.
When the delinquency exceeds 3 months (=3 installments, 3 payments missed), the loan is defined as defaulted.

This is the legal definition of defaults. When this happens, the bank must report to the relevant authority the default of the customer. 

For the purposes of this case:

- The loan amount can be considered lost when a default occurs.
- For portfolio management only the first 12 installments can be considered. Delinquencies on later installments are not provided, and hence can be ignored.

## Data Description

> Disclaimer: the data provided for the purpose of this exercise is in no way referred to actual bank clients, but it is rather based on simulations. Do not share this case or the dataset with others.

The data is added via [Git Large File Storage](https://git-lfs.github.com/) and should be in `data/`.
If you are having trouble accessing the data, first [install Git LFS](https://github.com/git-lfs/git-lfs/wiki/Installation) and then clone the repository.

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
- `joined_ING_date`: month and year when the customer became an ING client
- `number_client_calls_to_ING`: total number of times the customer has reached ING by phone
- `number_client_calls_from_ING`: number of times the customer has been contacted by phone by ING

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
 
