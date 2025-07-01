# Model training and evaluation functions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import os

def get_target_labels(df_with_labels, original_df):
    """
    Converts the 'real' label (1 or 2) into binary classification:
    1 if text_1 is real, 0 if text_2 is real
    """
    labels = (df_with_labels['real'] == 1).astype(int)
    return labels

def train_model(X, y, model_name='xgboost'):
    if model_name == 'xgboost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_name == 'lightgbm':
        model = LGBMClassifier(random_state=42)
    elif model_name == 'logistic':
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError("Unsupported model: " + model_name)

    model.fit(X, y)
    return model

def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def train_and_evaluate_all_models(features_df, raw_df, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)

    y = get_target_labels(raw_df, features_df)
    X = features_df.drop(columns=['id'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    for model_name in ['xgboost', 'lightgbm', 'logistic']:
        print(f"🔧 Training {model_name}...")
        model = train_model(X_train, y_train, model_name)
        acc = evaluate_model(model, X_val, y_val)
        save_model(model, os.path.join(save_dir, f"{model_name}_model.pkl"))
        results[model_name] = acc
        print(f"✅ {model_name} accuracy: {acc:.4f}")

    return results
