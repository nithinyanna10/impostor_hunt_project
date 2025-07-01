# Feature extraction methods (e.g., embeddings, similarity, readability)
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import textstat
import spacy

# Load once globally
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_sm')

def compute_embeddings(text):
    return model.encode([text])[0]

def extract_readability(text):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text)
    }

def count_named_entities(text):
    doc = nlp(text)
    return {
        "num_entities": len(doc.ents),
        "num_persons": sum(1 for ent in doc.ents if ent.label_ == "PERSON"),
        "num_orgs": sum(1 for ent in doc.ents if ent.label_ == "ORG"),
        "num_dates": sum(1 for ent in doc.ents if ent.label_ == "DATE")
    }

def extract_features(df):
    features = []

    for _, row in df.iterrows():
        t1 = row['text_1']
        t2 = row['text_2']
        emb1 = compute_embeddings(t1)
        emb2 = compute_embeddings(t2)

        cos_sim = cosine_similarity([emb1], [emb2])[0][0]

        feats = {
            "id": row["id"],
            "cosine_similarity": cos_sim,
            "len_text_1": len(t1),
            "len_text_2": len(t2),
            "len_diff": abs(len(t1) - len(t2)),
        }

        feats.update({f"t1_{k}": v for k, v in extract_readability(t1).items()})
        feats.update({f"t2_{k}": v for k, v in extract_readability(t2).items()})

        feats.update({f"t1_{k}": v for k, v in count_named_entities(t1).items()})
        feats.update({f"t2_{k}": v for k, v in count_named_entities(t2).items()})

        features.append(feats)

    return pd.DataFrame(features)
