import os
import joblib
import pandas as pd
from src.data_loader import load_test_data
from src.feature_engineering import extract_features
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Step 1: Load test data
test_df = load_test_data("data/test", verbose=True)
features_df = extract_features(test_df)
X_test = features_df.drop(columns=["id"])

# Output dir
os.makedirs("submission", exist_ok=True)

# Step 2: Predict with traditional models
model_names = {
    "xgboost": "xgboost_model_bert.pkl",
    "lightgbm": "lightgbm_model_bert.pkl",
    "logistic": "logistic_model_bert.pkl",
    "catboost": "catboost_model.pkl",
    "svm": "svm_model.pkl"
}

for name, file in model_names.items():
    model_path = os.path.join("models", file)
    model = joblib.load(model_path)

    # scale if logistic or pipeline
    if hasattr(model, "_scaler"):
        X = model._scaler.transform(X_test)
    else:
        X = X_test

    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    preds = (probs >= 0.5).astype(int)

    df_out = pd.DataFrame({
        "id": features_df["id"],
        "real": preds
    })
    out_path = f"submission/submission_{name}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"✅ Saved {name} predictions to {out_path}")

# Step 3: Predict with BERT model
def predict_with_bert(test_df, model_dir="models/bert_model"):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    predictions = []

    for _, row in test_df.iterrows():
        inputs = tokenizer(
            row['text_1'],
            row['text_2'],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=1).item()
            predictions.append(pred)

    df_out = pd.DataFrame({
        "id": test_df["id"],
        "real": predictions
    })
    df_out.to_csv("submission/submission_bert.csv", index=False)
    print("✅ BERT predictions saved to submission/submission_bert.csv")

# Run BERT prediction
predict_with_bert(test_df)
