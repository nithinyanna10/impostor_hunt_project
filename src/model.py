import os
import joblib
import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def get_target_labels(df_with_labels):
    return (df_with_labels['real'] == 1).astype(int)

def train_model(X, y, model_name='xgboost'):
    if model_name == 'xgboost':
        model = XGBClassifier(eval_metric='logloss', random_state=42)
    elif model_name == 'lightgbm':
        model = LGBMClassifier(random_state=42)
    elif model_name == 'logistic':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=5000, solver='saga')
        model._scaler = scaler
    else:
        raise ValueError("Unsupported model: " + model_name)

    model.fit(X, y)
    return model

def evaluate_model(model, X_val, y_val):
    if hasattr(model, "_scaler"):
        X_val = model._scaler.transform(X_val)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def train_and_evaluate_all_models(features_df, raw_df, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    y = get_target_labels(raw_df)
    X = features_df.drop(columns=['id'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    for model_name in ['xgboost', 'lightgbm', 'logistic']:
        print(f"🔧 Training {model_name}...")
        model = train_model(X_train, y_train, model_name)
        acc = evaluate_model(model, X_val, y_val)
        save_model(model, os.path.join(save_dir, f"{model_name}_model_bert.pkl"))
        results[model_name] = acc
        print(f"✅ {model_name} accuracy: {acc:.4f}")

    return results

def train_and_evaluate_more_models(features_df, raw_df):
    X = features_df.drop(columns=["id"])
    y = get_target_labels(raw_df)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # CatBoost
    print("🔧 Training CatBoost...")
    cat_model = CatBoostClassifier(verbose=0)
    cat_model.fit(X_train, y_train)
    acc_cat = accuracy_score(y_val, cat_model.predict(X_val))
    results["catboost"] = acc_cat
    print(f"✅ CatBoost accuracy: {acc_cat:.4f}")

    # SVM
    print("🔧 Training SVM...")
    svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True))
    svm_pipeline.fit(X_train, y_train)
    acc_svm = accuracy_score(y_val, svm_pipeline.predict(X_val))
    results["svm"] = acc_svm
    print(f"✅ SVM accuracy: {acc_svm:.4f}")

    return results

def train_bert_pair_model(raw_df, save_dir='models/bert_model'):
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data
    texts = []
    labels = []
    for _, row in raw_df.iterrows():
        texts.append((row["text_1"], row["text_2"]))
        labels.append(1 if row["real"] == 1 else 0)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    def tokenize(example):
        return tokenizer(example['text1'], example['text2'], truncation=True, padding='max_length', max_length=512)

    dataset = Dataset.from_dict({
        "text1": [t[0] for t in texts],
        "text2": [t[1] for t in texts],
        "label": labels
    }).train_test_split(test_size=0.2, seed=42)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    args = TrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=4,
    num_train_epochs=10,
    logging_dir='./logs',
    logging_steps=10
)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"]
    )

    print("🔧 Fine-tuning BERT pair model...")
    trainer.train()
    print("✅ BERT training complete. Saving model...")

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    return trainer.evaluate()["eval_loss"]
