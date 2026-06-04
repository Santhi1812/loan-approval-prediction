import numpy as np
import pandas as pd

def add_features(df):
    """
    Applies feature engineering to the dataset:
    - Calculates Total Income (ApplicantIncome + CoapplicantIncome)
    - Calculates Income-to-Loan Ratio
    - Performs log transformation on highly skewed numerical features:
      ApplicantIncome, LoanAmount, and Total_Income.
    """
    data = df.copy()
    
    # 1. Total Income
    data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    
    # 2. Income-to-Loan Ratio (adding 1 to LoanAmount to prevent division by zero/nan)
    data['Income_to_Loan_Ratio'] = data['Total_Income'] / (data['LoanAmount'] + 1.0)
    
    # 3. Log Transformations for Skewness reduction
    data['ApplicantIncome_Log'] = np.log1p(data['ApplicantIncome'])
    data['LoanAmount_Log'] = np.log1p(data['LoanAmount'])
    data['Total_Income_Log'] = np.log1p(data['Total_Income'])
    
    return data
