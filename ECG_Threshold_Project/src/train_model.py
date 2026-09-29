import os
import sys
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_ptbxl_metadata, load_ecg_data
from src.labels import create_binary_labels
from src.preprocessing import preprocess_signals
from src.feature_extraction import extract_features
from src.evaluation import evaluate_predictions, print_evaluation
from src.threshold_analysis import run_threshold_experiment, plot_roc_curve

def main():
    print("--- ECG Classification Training Pipeline ---")
    data_dir = 'data/ptb-xl/'
    
    # 1. Load metadata
    print("Loading metadata...")
    df = load_ptbxl_metadata(data_dir=data_dir, max_records=5000)
    if df is None:
        print("PTB-XL dataset is required before model training can be executed.")
        sys.exit(1)
        
    # 2. Create binary labels
    print("Creating binary labels...")
    df = create_binary_labels(df, data_dir=data_dir)
    if df.empty:
        print("No valid records found after labeling. Exiting.")
        sys.exit(1)
        
    # 3. Load ECG data
    print("Loading ECG waveforms...")
    X, df_valid = load_ecg_data(df, sampling_rate=100, data_dir=data_dir)
    if len(X) == 0:
        print("Failed to load ECG signals. Exiting.")
        sys.exit(1)
        
    y = df_valid['label'].values
    
    # 4. Preprocess signals
    print("Preprocessing signals...")
    X_preprocessed = preprocess_signals(X)
    
    # 5. Extract features
    print("Extracting features...")
    X_features = extract_features(X_preprocessed)
    
    # 6. Train/Test split
    print("Splitting dataset (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # 7. Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 8. Train Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # 9. Get probabilities
    print("Generating predictions on test set...")
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # 10. Baseline Evaluation
    print("\nEvaluating Baseline Model (Threshold = 0.50):")
    baseline_metrics = evaluate_predictions(y_test, y_probs, threshold=0.50)
    print_evaluation(baseline_metrics)
    
    # 11. Decision Threshold Experiment
    print("\nRunning Decision Threshold Experiment...")
    thresholds_to_test = [0.20, 0.40, 0.50, 0.60, 0.80]
    df_results = run_threshold_experiment(y_test, y_probs, thresholds=thresholds_to_test, results_dir='results/')
    print(df_results.to_string(index=False))
    
    # 12. ROC Curve
    print("Generating ROC Curve...")
    roc_auc = plot_roc_curve(y_test, y_probs, results_dir='results/')
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # 13. Save model and scaler
    print("\nSaving model and scaler...")
    os.makedirs('model/', exist_ok=True)
    joblib.dump(model, 'model/ecg_model.pkl')
    joblib.dump(scaler, 'model/ecg_scaler.pkl')
    
    # Save feature configuration for reproducibility
    config = {
        "features": ["mean", "std", "min", "max", "rms", "range"],
        "num_leads": 12,
        "features_per_lead": 6
    }
    with open('model/feature_config.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    print("Model, scaler, and configuration saved to model/ directory.")
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
