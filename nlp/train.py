"""
Model Training Module for AI Chatbot.
Loads conversational training patterns from JSON dataset, preprocesses text,
trains a TF-IDF + Logistic Regression pipeline, and serializes the model.
"""

import json
import os
import sys
from typing import Tuple, List, Dict, Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Support running directly or as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nlp.preprocessing import preprocess_text
from nlp.evaluation import evaluate_model


def load_dataset(dataset_path: str = "data/intents.json") -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Load intents dataset and extract raw patterns and intent tags."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    patterns = []
    labels = []
    
    for intent in data.get("intents", []):
        tag = intent["tag"]
        for pattern in intent.get("patterns", []):
            patterns.append(pattern)
            labels.append(tag)
            
    return patterns, labels, data


def build_pipeline() -> Pipeline:
    """Construct Scikit-learn Pipeline with TF-IDF Vectorizer and Logistic Regression."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents='unicode'
        )),
        ('classifier', LogisticRegression(
            C=12.0,
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        ))
    ])


def train_and_evaluate(
    dataset_path: str = "data/intents.json",
    model_dir: str = "models",
    results_dir: str = "results",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Main training workflow:
    1. Load intents
    2. Preprocess text patterns
    3. Split into train & test sets (stratified)
    4. Fit TF-IDF and Logistic Regression
    5. Save model package
    6. Evaluate and save reports and confusion matrix
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"[*] Loading training data from: {dataset_path}")
    raw_patterns, labels, intents_data = load_dataset(dataset_path)
    print(f"[*] Total training patterns: {len(raw_patterns)} across {len(set(labels))} intents")
    
    # Preprocess patterns
    print("[*] Preprocessing conversational patterns...")
    clean_patterns = [preprocess_text(p) for p in raw_patterns]
    
    # Train / Test Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        clean_patterns,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    print(f"[*] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    
    # Train pipeline
    print("[*] Training TF-IDF + Logistic Regression Pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Classes
    classes = list(pipeline.named_steps['classifier'].classes_)
    
    # Save model artifact
    model_artifact = {
        'pipeline': pipeline,
        'intents_data': intents_data,
        'classes': classes
    }
    model_path = os.path.join(model_dir, "chatbot_model.pkl")
    joblib.dump(model_artifact, model_path)
    print(f"[✓] Model artifact successfully saved to: {model_path}")
    
    # Evaluate model
    print("[*] Evaluating model and generating confusion matrix...")
    metrics = evaluate_model(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        classes=classes,
        results_dir=results_dir
    )
    print(f"[✓] Evaluation completed!")
    print(f"    - Overall Accuracy:   {metrics['accuracy'] * 100:.2f}%")
    print(f"    - Macro F1-Score:     {metrics['f1_macro'] * 100:.2f}%")
    print(f"    - Weighted F1-Score:  {metrics['f1_weighted'] * 100:.2f}%")
    print(f"    - Confusion Matrix:   {os.path.join(results_dir, 'confusion_matrix.png')}")
    print(f"    - Evaluation Report:  {os.path.join(results_dir, 'evaluation_report.txt')}")
    
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
