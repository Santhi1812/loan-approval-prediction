import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Import modules from src
from data_preprocessing import load_data, preprocess_data
from feature_engineering import add_features

def main():
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    print("Loading dataset...")
    df_raw = load_data("data/LoanApprovalPrediction.csv")
    print(f"Dataset shape: {df_raw.shape}")
    
    print("\nPreprocessing data...")
    df_preprocessed, encoders, impute_values = preprocess_data(df_raw, is_training=True)
    
    print("\nEngineering features...")
    df_features = add_features(df_preprocessed)
    
    # Save preprocessing artifacts for inference
    artifacts = {
        'encoders': encoders,
        'impute_values': impute_values,
        'feature_cols': [c for c in df_features.columns if c != 'Loan_Status']
    }
    joblib.dump(artifacts, "models/preprocessing_artifacts.joblib")
    print("Saved preprocessing artifacts to models/preprocessing_artifacts.joblib")
    
    # Split into features (X) and target (y)
    X = df_features.drop(['Loan_Status'], axis=1)
    y = df_features['Loan_Status']
    
    # Split into train and test sets (60-40 split, random_state=1 as per tutorial)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Initialize models
    # Random Forest setup matching the tutorial's hyperparameters
    rfc = RandomForestClassifier(
        n_estimators=7, criterion='entropy', random_state=7
    )
    # Logistic Regression with increased max_iter for convergence
    lr = LogisticRegression(max_iter=1000, random_state=1)
    
    # Fit and evaluate models
    models = {
        'RandomForestClassifier': rfc,
        'LogisticRegression': lr
    }
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        
        # Predict on train
        y_train_pred = model.predict(X_train)
        train_acc = metrics.accuracy_score(y_train, y_train_pred) * 100
        
        # Predict on test
        y_test_pred = model.predict(X_test)
        test_acc = metrics.accuracy_score(y_test, y_test_pred) * 100
        
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Testing Accuracy: {test_acc:.2f}%")
        
        # Save trained model
        if name == 'LogisticRegression':
            model_filename = "models/logistic_regression.joblib"
        elif name == 'RandomForestClassifier':
            model_filename = "models/random_forest.joblib"
        else:
            model_filename = f"models/{name.lower()}.joblib"
        joblib.dump(model, model_filename)
        print(f"Saved model to {model_filename}")

if __name__ == "__main__":
    main()
