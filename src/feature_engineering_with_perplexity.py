
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import textstat
import spacy
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

# Load models globally
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_sm')
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_model.eval()

def compute_embeddings(text):
    return model.encode([text])[0]

def get_perplexity(text):
    try:
        encodings = gpt2_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = gpt2_model(**encodings, labels=encodings['input_ids'])
        loss = outputs.loss
        return torch.exp(loss).item()
    except Exception as e:
        print(f"⚠️ Perplexity error: {e}")
        return -1.0

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

        # Compute BERT embeddings
        emb1 = compute_embeddings(t1)
        emb2 = compute_embeddings(t2)

        # Compute GPT-2 perplexity ONCE for each
        perplexity_1 = get_perplexity(t1)
        perplexity_2 = get_perplexity(t2)
        perplexity_diff = abs(perplexity_1 - perplexity_2)
        print(f"✅ ID {row.get('id', -1)} | P1: {perplexity_1:.2f} | P2: {perplexity_2:.2f} | Δ: {perplexity_diff:.2f}")

        # Cosine sim
        cos_sim = cosine_similarity([emb1], [emb2])[0][0]

        feats = {
            "id": row.get("id", -1),
            "cosine_similarity": cos_sim,
            "len_text_1": len(t1),
            "len_text_2": len(t2),
            "len_diff": abs(len(t1) - len(t2)),
            "perplexity_1": perplexity_1,
            "perplexity_2": perplexity_2,
            "perplexity_diff": perplexity_diff
        }

        # Embedding interactions
        feats.update({f"emb_diff_{i}": emb1[i] - emb2[i] for i in range(len(emb1))})
        feats.update({f"emb_prod_{i}": emb1[i] * emb2[i] for i in range(len(emb1))})
        feats.update({f"emb_concat_1_{i}": emb1[i] for i in range(len(emb1))})
        feats.update({f"emb_concat_2_{i}": emb2[i] for i in range(len(emb2))})

        # Readability
        feats.update({f"t1_{k}": v for k, v in extract_readability(t1).items()})
        feats.update({f"t2_{k}": v for k, v in extract_readability(t2).items()})

        # NER
        feats.update({f"t1_{k}": v for k, v in count_named_entities(t1).items()})
        feats.update({f"t2_{k}": v for k, v in count_named_entities(t2).items()})

        features.append(feats)

    return pd.DataFrame(features)
