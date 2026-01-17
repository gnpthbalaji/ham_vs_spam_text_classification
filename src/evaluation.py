
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
from typing import Dict, Any, Tuple


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> Dict[str, float]:
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    return metrics


def print_evaluation_report(y_true, y_pred, model_name: str = "Model"):
    print(f"\n{'='*60}")
    print(f"Evaluation Report: {model_name}")
    print(f"{'='*60}\n")
    
    metrics = evaluate_model(y_true, y_pred, model_name)
    
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    
    print(f"\n{'-'*60}")
    print("Classification Report:")
    print(f"{'-'*60}\n")
    print(classification_report(y_true, y_pred, target_names=['ham', 'spam']))


def plot_confusion_matrix(y_true, y_pred, 
                         model_name: str = "Model",
                         save_path: str = None,
                         figsize: Tuple[int, int] = (8, 6)):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                ax=ax)
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix - {model_name}')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig, ax


def analyze_errors(y_true, y_pred, X_text, 
                   n_samples: int = 5) -> Dict[str, list]:
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'text': X_text,
        'true_label': y_true,
        'predicted_label': y_pred
    })
    
    # False positives (predicted spam, actually ham)
    false_positives = df[(df['true_label'] == 0) & (df['predicted_label'] == 1)]
    
    # False negatives (predicted ham, actually spam)
    false_negatives = df[(df['true_label'] == 1) & (df['predicted_label'] == 0)]
    
    results = {
        'false_positives': false_positives['text'].head(n_samples).tolist(),
        'false_negatives': false_negatives['text'].head(n_samples).tolist(),
        'fp_count': len(false_positives),
        'fn_count': len(false_negatives)
    }
    
    return results


def print_error_analysis(errors: Dict[str, Any]):
    print(f"\n{'='*60}")
    print("Error Analysis")
    print(f"{'='*60}\n")
    
    print(f"Total False Positives (Ham predicted as Spam): {errors['fp_count']}")
    print(f"Total False Negatives (Spam predicted as Ham): {errors['fn_count']}")
    
    print(f"\n{'-'*60}")
    print("Sample False Positives (Ham messages classified as Spam):")
    print(f"{'-'*60}")
    for i, text in enumerate(errors['false_positives'], 1):
        print(f"{i}. {text}")
    
    print(f"\n{'-'*60}")
    print("Sample False Negatives (Spam messages classified as Ham):")
    print(f"{'-'*60}")
    for i, text in enumerate(errors['false_negatives'], 1):
        print(f"{i}. {text}")


def plot_roc_curve(y_true, y_pred_proba,
                  model_name: str = "Model",
                  save_path: str = None,
                  figsize: Tuple[int, int] = (8, 6)):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    return fig, ax, auc_score


def compare_models(results_list: list) -> pd.DataFrame:
    df = pd.DataFrame(results_list)
    df = df.round(4)
    return df


def save_metrics_to_csv(metrics: Dict[str, float], filepath: str):
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")


def plot_class_distribution(y, 
                           title: str = "Class Distribution",
                           save_path: str = None,
                           figsize: Tuple[int, int] = (8, 6)):
    # Count values
    counts = pd.Series(y).value_counts().sort_index()
    labels = ['Ham', 'Spam']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    bars = ax.bar(labels, counts.values, color=['green', 'red'], alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)} ({height/sum(counts.values)*100:.1f}%)',
                ha='center', va='bottom')
    
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    return fig, ax


def plot_message_length_distribution(df: pd.DataFrame,
                                    length_col: str = 'message_length',
                                    label_col: str = 'label',
                                    save_path: str = None,
                                    figsize: Tuple[int, int] = (10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot distributions
    for label in df[label_col].unique():
        data = df[df[label_col] == label][length_col]
        ax.hist(data, bins=50, alpha=0.6, label=label.capitalize())
    
    ax.set_xlabel('Message Length (characters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Message Length Distribution by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Message length distribution saved to {save_path}")
    
    return fig, ax