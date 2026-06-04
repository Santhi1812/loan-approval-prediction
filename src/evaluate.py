import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Import modules from src
from data_preprocessing import load_data, preprocess_data
from feature_engineering import add_features

def main():
    print("Loading test data...")
    # Load raw data and re-run preprocessing and splitting to isolate the exact test set
    df_raw = load_data("data/LoanApprovalPrediction.csv")
    df_preprocessed, _, _ = preprocess_data(df_raw, is_training=True)
    df_features = add_features(df_preprocessed)
    
    X = df_features.drop(['Loan_Status'], axis=1)
    y = df_features['Loan_Status']
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1
    )
    
    # Load preprocessing artifacts to get target encoder classes
    try:
        artifacts = joblib.load("models/preprocessing_artifacts.joblib")
        target_encoder = artifacts['encoders']['Loan_Status']
        target_names = list(target_encoder.classes_)
    except Exception as e:
        print(f"Warning: Could not load class names from encoders. Error: {e}")
        target_names = ['N', 'Y']
        
    models = {
        'Random Forest': 'models/random_forest.joblib',
        'Logistic Regression': 'models/logistic_regression.joblib'
    }
    
    for name, filepath in models.items():
        print(f"\n=========================================")
        print(f" Evaluating: {name}")
        print(f"=========================================")
        
        try:
            model = joblib.load(filepath)
        except FileNotFoundError:
            print(f"Error: Trained model file {filepath} not found. Please run src/train.py first.")
            continue
            
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        acc = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        
        print(f"Accuracy:  {acc * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall:    {recall * 100:.2f}%")
        print(f"F1-Score:  {f1 * 100:.2f}%")
        
        if y_proba is not None:
            roc_auc = metrics.roc_auc_score(y_test, y_proba)
            print(f"ROC AUC:   {roc_auc * 100:.2f}%")
            
        print("\nClassification Report:")
        print(metrics.classification_report(y_test, y_pred, target_names=target_names))
        
        print("Confusion Matrix:")
        cm = metrics.confusion_matrix(y_test, y_pred)
        print(f"               Predicted {target_names[0]}   Predicted {target_names[1]}")
        print(f"Actual {target_names[0]:<10} {cm[0][0]:<14} {cm[0][1]:<14}")
        print(f"Actual {target_names[1]:<10} {cm[1][0]:<14} {cm[1][1]:<14}")

if __name__ == "__main__":
    main()
