import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""# 1. Project Introduction
This project investigates the effect of changing decision thresholds on the performance of a Logistic Regression model for ECG classification.

# 2. Team Information
**Team 03**
- Abhijith P
- Noel Biju
- Diya Krishna
- Ann Mariya Biju

# 3. Problem Statement
Default machine learning models typically use a 0.50 threshold for binary classification. In medical contexts, this may not be optimal depending on the cost of false positives vs false negatives.

# 4. Research Question
How does changing the decision threshold affect the behaviour and performance of an ECG classification model?

# 5. Dataset Description
We use the PTB-XL ECG dataset from PhysioNet, containing clinical ECG recordings and diagnostic statements.
"""))

cells.append(nbf.v4.new_markdown_cell("# 6. Imports"))
cells.append(nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.append(os.path.abspath('..'))
from src.data_loader import load_ptbxl_metadata, load_ecg_data
from src.labels import create_binary_labels
from src.preprocessing import preprocess_signals
from src.feature_extraction import extract_features
from src.evaluation import evaluate_predictions, print_evaluation
from src.threshold_analysis import run_threshold_experiment, plot_roc_curve
"""))

cells.append(nbf.v4.new_markdown_cell("# 7. Configuration"))
cells.append(nbf.v4.new_code_cell("""DATA_DIR = '../data/ptb-xl/'
RESULTS_DIR = '../results/'
MAX_RECORDS = 5000
RANDOM_STATE = 42
"""))

cells.append(nbf.v4.new_markdown_cell("# 8. Load Dataset"))
cells.append(nbf.v4.new_code_cell("""df_meta = load_ptbxl_metadata(data_dir=DATA_DIR, max_records=MAX_RECORDS)
if df_meta is None:
    print("PTB-XL dataset is missing. Please download it and place it in the data/ptb-xl/ folder.")
else:
    print(f"Loaded {len(df_meta)} records.")
"""))

cells.append(nbf.v4.new_markdown_cell("# 9. Explore Dataset"))
cells.append(nbf.v4.new_code_cell("""if df_meta is not None:
    display(df_meta.head())
"""))

cells.append(nbf.v4.new_markdown_cell("# 10. Create Binary Labels\n# 11. Check Class Distribution"))
cells.append(nbf.v4.new_code_cell("""if df_meta is not None:
    df_meta = create_binary_labels(df_meta, data_dir=DATA_DIR)
"""))

cells.append(nbf.v4.new_markdown_cell("# 12. ECG Preprocessing\nFirst, load the actual ECG signals."))
cells.append(nbf.v4.new_code_cell("""if df_meta is not None:
    X, df_valid = load_ecg_data(df_meta, sampling_rate=100, data_dir=DATA_DIR)
    y = df_valid['label'].values
    
    print("Preprocessing signals...")
    X_preprocessed = preprocess_signals(X)
    print(f"Signal shape: {X_preprocessed.shape}")
"""))

cells.append(nbf.v4.new_markdown_cell("# 13. Feature Extraction"))
cells.append(nbf.v4.new_code_cell("""if df_meta is not None:
    print("Extracting features...")
    X_features = extract_features(X_preprocessed)
    print(f"Features shape: {X_features.shape}")
"""))

cells.append(nbf.v4.new_markdown_cell("# 14. Train/Test Split"))
cells.append(nbf.v4.new_code_cell("""if df_meta is not None:
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
"""))

cells.append(nbf.v4.new_markdown_cell("# 15. Feature Scaling"))
cells.append(nbf.v4.new_code_cell("""if df_meta is not None:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
"""))

cells.append(nbf.v4.new_markdown_cell("# 16. Train Logistic Regression"))
cells.append(nbf.v4.new_code_cell("""if df_meta is not None:
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    print("Model trained.")
"""))

cells.append(nbf.v4.new_markdown_cell("# 17. Baseline Threshold = 0.50"))
cells.append(nbf.v4.new_code_cell("""if df_meta is not None:
    baseline_metrics = evaluate_predictions(y_test, y_probs, threshold=0.50)
    print_evaluation(baseline_metrics)
"""))

cells.append(nbf.v4.new_markdown_cell("# 18. Decision Threshold Experiment\n# 19. Metrics Table\n# 20. Confusion Matrices\n# 21. Threshold Graphs"))
cells.append(nbf.v4.new_code_cell("""if df_meta is not None:
    thresholds_to_test = [0.20, 0.40, 0.50, 0.60, 0.80]
    df_results = run_threshold_experiment(y_test, y_probs, thresholds=thresholds_to_test, results_dir=RESULTS_DIR)
    display(df_results)
"""))

cells.append(nbf.v4.new_markdown_cell("# 22. ROC Curve"))
cells.append(nbf.v4.new_code_cell("""if df_meta is not None:
    roc_auc = plot_roc_curve(y_test, y_probs, results_dir=RESULTS_DIR)
    print(f"ROC-AUC: {roc_auc:.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""# 23. Results Discussion
By lowering the decision threshold, we increase Recall (reducing False Negatives) but decrease Precision (increasing False Positives). A higher threshold does the opposite. In an ECG classification context, we typically prefer fewer false negatives.

# 24. Limitations
- We are using a simplified feature set (basic statistical features) instead of deep learning (e.g., CNNs/ResNets) which are state-of-the-art for raw ECG waveforms.
- The threshold selection is evaluated on the test set directly for demonstration, whereas in a strict ML pipeline it should be done on a separate validation set.

# 25. Conclusion
Changing the decision threshold drastically alters the model's behaviour. Our experiment shows how a standard threshold of 0.50 is not always optimal and must be tuned according to the specific priorities of the classification task.
"""))

nb['cells'] = cells

with open('notebooks/ECG_Threshold_Investigation.ipynb', 'w') as f:
    nbf.write(nb, f)
