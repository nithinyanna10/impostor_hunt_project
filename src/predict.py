import pandas as pd
import numpy as np
import os
from src.feature_engineering import extract_features
from src.data_loader import load_train_data
from src.model import load_model

def predict_real_file_id(proba_preds):
    """
    Convert prediction: if prob > 0.5, text_1 is real → return 1
    else → return 2 (text_2 is real)
    """
    return [1 if p > 0.5 else 2 for p in proba_preds]

def predict_on_test(model_path, raw_test_df, output_csv_path):
    model = load_model(model_path)
    features_df = extract_features(raw_test_df)
    X = features_df.drop(columns=['id'])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)

    predictions = predict_real_file_id(probs)

    submission_df = pd.DataFrame({
        'id': raw_test_df['id'],
        'real_text_id': predictions
    })

    submission_df.to_csv(output_csv_path, index=False)
    print(f"✅ Saved predictions to {output_csv_path}")
