"""
Model Evaluation and Metrics Visualization Module.
Evaluates the trained intent classification model, computes key statistical metrics,
and produces academic-grade evaluation visualizations and text reports.
"""

import json
import os
from typing import Dict, Any, List

import matplotlib
matplotlib.use('Agg')  # Headless backend for server/script execution
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_model(
    pipeline,
    X_test: List[str],
    y_test: List[str],
    classes: List[str],
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Perform comprehensive model evaluation on the test set.
    Generates metrics.json, evaluation_report.txt, and confusion_matrix.png.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    
    precision_weighted = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recall_weighted = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    
    report_text = classification_report(y_test, y_pred, zero_division=0)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(precision_weighted, 4),
        "recall_weighted": round(recall_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "test_sample_count": len(y_test),
        "number_of_classes": len(classes),
        "classification_report": report_dict
    }
    
    # 1. Save metrics.json
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    # 2. Save evaluation_report.txt
    report_path = os.path.join(results_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(" " * 15 + "AI CHATBOT MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Algorithm:           TF-IDF Vectorizer + Logistic Regression\n")
        f.write(f"Total Test Samples:  {len(y_test)}\n")
        f.write(f"Number of Intents:   {len(classes)}\n\n")
        f.write("-" * 70 + "\n")
        f.write("SUMMARY PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Overall Accuracy:        {accuracy * 100:.2f}%\n")
        f.write(f"Macro Precision:         {precision_macro * 100:.2f}%\n")
        f.write(f"Macro Recall:            {recall_macro * 100:.2f}%\n")
        f.write(f"Macro F1-Score:          {f1_macro * 100:.2f}%\n")
        f.write(f"Weighted Precision:      {precision_weighted * 100:.2f}%\n")
        f.write(f"Weighted Recall:         {recall_weighted * 100:.2f}%\n")
        f.write(f"Weighted F1-Score:       {f1_weighted * 100:.2f}%\n\n")
        f.write("-" * 70 + "\n")
        f.write("DETAILED PER-INTENT CLASSIFICATION REPORT\n")
        f.write("-" * 70 + "\n")
        f.write(report_text + "\n")
        f.write("=" * 70 + "\n")
        
    # 3. Plot Confusion Matrix
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(14, 12), facecolor="#101827")
    ax.set_facecolor("#1E293B")
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=True
    )
    
    # Customizing fonts and colors to match Dark UI theme
    ax.set_title("Intent Classification Confusion Matrix", color="#F8FAFC", fontsize=16, pad=15, weight='bold')
    ax.set_xlabel("Predicted Intent", color="#F8FAFC", fontsize=12, labelpad=10)
    ax.set_ylabel("True Intent", color="#F8FAFC", fontsize=12, labelpad=10)
    ax.tick_params(colors="#F8FAFC", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Customize colorbar text color
    cbar = disp.ax_.images[0].colorbar
    if cbar:
        cbar.ax.yaxis.set_tick_params(color="#F8FAFC")
        plt.setp(cbar.ax.get_yticklabels(), color="#F8FAFC")
    
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    
    return metrics
