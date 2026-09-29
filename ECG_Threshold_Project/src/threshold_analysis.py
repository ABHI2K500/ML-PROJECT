import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from .evaluation import evaluate_predictions

def run_threshold_experiment(y_true, y_probs, thresholds, results_dir='results/'):
    """
    Evaluates multiple thresholds and saves the results.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    
    for t in thresholds:
        metrics = evaluate_predictions(y_true, y_probs, threshold=t)
        results.append(metrics)
        
        # Save confusion matrix plot
        plot_confusion_matrix(metrics['Confusion_Matrix'], t, results_dir)
        
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save CSV
    csv_path = os.path.join(results_dir, 'threshold_results.csv')
    df_results[['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1', 'FP', 'FN', 'TP', 'TN']].to_csv(csv_path, index=False)
    
    # Generate graphs
    plot_threshold_metrics(df_results, results_dir)
    plot_threshold_fp_fn(df_results, results_dir)
    
    return df_results

def plot_confusion_matrix(cm, threshold, results_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    filename = f'confusion_matrix_{int(threshold * 100):03d}.png'
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def plot_threshold_metrics(df, results_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Threshold'], df['Precision'], marker='o', label='Precision')
    plt.plot(df['Threshold'], df['Recall'], marker='s', label='Recall')
    plt.plot(df['Threshold'], df['F1'], marker='^', label='F1-score')
    plt.plot(df['Threshold'], df['Accuracy'], marker='x', label='Accuracy')
    
    plt.title('Threshold vs Classification Metrics')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'threshold_metrics.png'))
    plt.close()

def plot_threshold_fp_fn(df, results_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Threshold'], df['FP'], marker='o', color='red', label='False Positives')
    plt.plot(df['Threshold'], df['FN'], marker='s', color='orange', label='False Negatives')
    
    plt.title('Threshold vs False Positives & False Negatives')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'threshold_fp_fn.png'))
    plt.close()

def plot_roc_curve(y_true, y_probs, results_dir='results/'):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.close()
    
    return roc_auc
