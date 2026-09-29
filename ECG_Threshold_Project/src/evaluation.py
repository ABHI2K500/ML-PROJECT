import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_predictions(y_true, y_probs, threshold=0.5):
    """
    Evaluates predictions at a given threshold.
    
    Parameters:
    - y_true: array-like, true binary labels
    - y_probs: array-like, predicted probabilities for class 1
    - threshold: float, decision threshold
    
    Returns:
    - dict: dictionary containing all metrics
    """
    y_pred = (y_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    # Use zero_division=0 to prevent warnings when no positive predictions are made
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case handling if only one class is present in y_true or y_pred
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_true)) == 1:
            if y_true.iloc[0] if hasattr(y_true, 'iloc') else y_true[0] == 0:
                tn = cm[0,0]
            else:
                tp = cm[0,0]
                
    metrics = {
        'Threshold': threshold,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Confusion_Matrix': cm
    }
    
    return metrics

def print_evaluation(metrics):
    """
    Prints the evaluation metrics clearly.
    """
    print(f"--- Evaluation at Threshold = {metrics['Threshold']:.2f} ---")
    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1-score:  {metrics['F1']:.4f}")
    print("Confusion Matrix:")
    print(f"TN: {metrics['TN']} | FP: {metrics['FP']}")
    print(f"FN: {metrics['FN']} | TP: {metrics['TP']}")
    print("-" * 40)
