import os
import joblib
import pandas as pd
import numpy as np

# Import modules from src
# Using relative imports or adding to path if needed, but since we run from parent or app, we can just import.
# Note: we will add src to Python path or handle it elegantly.
from data_preprocessing import preprocess_data
from feature_engineering import add_features

def predict_loan(input_dict, model_type='random_forest'):
    """
    Predicts the loan approval status for a single applicant profile.
    
    Parameters:
    -----------
    input_dict : dict
        A dictionary containing keys:
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area'
    model_type : str
        'random_forest' or 'logistic_regression'
        
    Returns:
    --------
    dict
        A dictionary containing:
        'prediction': 'Y' or 'N' (encoded values converted back)
        'status': 'Approved' or 'Rejected'
        'probability': float (probability of loan approval)
    """
    # 1. Convert single input dictionary to DataFrame
    df_raw = pd.DataFrame([input_dict])
    
    # 2. Load preprocessing artifacts
    artifacts_path = "models/preprocessing_artifacts.joblib"
    if not os.path.exists(artifacts_path):
        # Check if running from app directory
        artifacts_path = os.path.join("..", artifacts_path)
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError("Preprocessing artifacts not found. Please run training first.")
            
    artifacts = joblib.load(artifacts_path)
    encoders = artifacts['encoders']
    impute_values = artifacts['impute_values']
    feature_cols = artifacts['feature_cols']
    
    # Ensure all expected columns exist in input, default to NaN if missing
    expected_raw_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                         'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                         'Credit_History', 'Property_Area']
    for col in expected_raw_cols:
        if col not in df_raw.columns:
            df_raw[col] = np.nan
            
    # 3. Apply preprocessing (inference mode)
    df_preprocessed = preprocess_data(df_raw, is_training=False, encoders=encoders, impute_values=impute_values)
    
    # 4. Apply feature engineering
    df_features = add_features(df_preprocessed)
    
    # 5. Reorder columns to match training set exactly
    df_model_input = df_features[feature_cols]
    
    # 6. Load selected model
    model_filename = f"{model_type.lower()}.joblib"
    model_path = os.path.join("models", model_filename)
    if not os.path.exists(model_path):
        model_path = os.path.join("..", "models", model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_filename} not found.")
            
    model = joblib.load(model_path)
    
    # 7. Make prediction
    pred_code = model.predict(df_model_input)[0]
    probabilities = model.predict_proba(df_model_input)[0]
    
    # Translate target code back to original label
    target_encoder = encoders['Loan_Status']
    prediction_label = target_encoder.inverse_transform([pred_code])[0]
    
    # The probability of 'Y' (Approval)
    # Target encoder classes are sorted, so 'N' is index 0, 'Y' is index 1
    y_class_idx = list(target_encoder.classes_).index('Y')
    approval_prob = probabilities[y_class_idx]
    
    return {
        'prediction': prediction_label,
        'status': 'Approved' if prediction_label == 'Y' else 'Rejected',
        'probability': float(approval_prob)
    }

if __name__ == "__main__":
    # Test prediction
    test_input = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': 2.0,
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150.0,
        'Loan_Amount_Term': 360.0,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }
    
    try:
        print("Running test prediction...")
        result = predict_loan(test_input, 'random_forest')
        print(f"Result: {result}")
    except Exception as e:
        print(f"Could not run test prediction. Have models been trained? Error: {e}")
