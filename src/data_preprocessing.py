import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """
    Loads the Loan Approval Prediction dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(df, is_training=True, encoders=None, impute_values=None):
    """
    Preprocesses the Loan Approval dataset:
    - Drops Loan_ID if present.
    - Encodes categorical variables.
    - Imputes missing values using training set column means.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The raw dataset.
    is_training : bool
        If True, fits new label encoders and calculates column means.
        If False, applies existing encoders and imputes using training means.
    encoders : dict, optional
        Fitted LabelEncoders (required if is_training=False).
    impute_values : dict, optional
        Column mean imputation values (required if is_training=False).
        
    Returns:
    --------
    If is_training is True:
        (processed_df, encoders, impute_values)
    If is_training is False:
        processed_df
    """
    # Create a copy to avoid SettingWithCopyWarning
    data = df.copy()
    
    # Drop Loan_ID if present
    if 'Loan_ID' in data.columns:
        data.drop(['Loan_ID'], axis=1, inplace=True)
        
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    target_col = 'Loan_Status'
    
    if is_training:
        encoders = {}
        impute_values = {}
        
        # 1. Label encode categorical variables
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                encoders[col] = le
            
        if target_col in data.columns:
            le_target = LabelEncoder()
            data[target_col] = le_target.fit_transform(data[target_col].astype(str))
            encoders[target_col] = le_target
            
        # 2. Impute missing values with column means (matching the tutorial's fillna logic)
        for col in data.columns:
            # We calculate mean of each column. Since we label-encoded object columns,
            # they are now numeric and mean can be computed safely.
            mean_val = data[col].mean()
            data[col] = data[col].fillna(mean_val)
            impute_values[col] = mean_val
            
        return data, encoders, impute_values
    else:
        if encoders is None or impute_values is None:
            raise ValueError("encoders and impute_values must be provided for test/inference preprocessing.")
            
        # 1. Label encode categorical variables using loaded encoders
        for col in categorical_cols:
            if col in data.columns:
                le = encoders[col]
                # Map unseen labels to the first class or a default value
                data[col] = data[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
                data[col] = le.transform(data[col])
                
        if target_col in data.columns and target_col in encoders:
            le_target = encoders[target_col]
            data[target_col] = data[target_col].astype(str).map(lambda s: s if s in le_target.classes_ else le_target.classes_[0])
            data[target_col] = le_target.transform(data[target_col])
            
        # 2. Impute missing values using training-set computed means
        for col in data.columns:
            if col in impute_values:
                data[col] = data[col].fillna(impute_values[col])
            else:
                data[col] = data[col].fillna(0.0)
                
        return data
