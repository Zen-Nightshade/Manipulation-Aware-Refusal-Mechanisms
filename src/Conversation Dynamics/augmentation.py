import random
import torch
import numpy as np
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from .config.paths import MAJ_P, MAJ_R, MAJ_A, CON_R, CON_P, CON_A


# -----------------------------
# LOAD MODELS
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

en_de_model_name = "Helsinki-NLP/opus-mt-en-de"
de_en_model_name = "Helsinki-NLP/opus-mt-de-en"

en_de_tokenizer = MarianTokenizer.from_pretrained(en_de_model_name)
en_de_model = MarianMTModel.from_pretrained(en_de_model_name).to(device)

de_en_tokenizer = MarianTokenizer.from_pretrained(de_en_model_name)
de_en_model = MarianMTModel.from_pretrained(de_en_model_name).to(device)

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# BACK TRANSLATION
# -----------------------------
def translate(text, tokenizer, model):
    batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        translated = model.generate(**batch)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def back_translate(text):
    try:
        de = translate(text, en_de_tokenizer, en_de_model)
        en = translate(de, de_en_tokenizer, de_en_model)
        return en
    except:
        return text


# -----------------------------
# SIMILARITY FILTER
# -----------------------------
def is_valid_aug(original, augmented, threshold=0.85):
    emb = embedder.encode([original, augmented])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return sim >= threshold


# -----------------------------
# TARGET DISTRIBUTION LOGIC
# -----------------------------
TARGET_MULTIPLIER = {
    "denial": 1.5,
    "evasion": 1.5,
    "feigning innocence": 2.0,
    "rationalization": 1.0,
    "playing victim role": 1.5,
    "playing servant role": 3.0,   # very low → heavily boost
    "shaming or belittlement": 1.0,
    "intimidation": 1.0,
    "brandishing anger": 1.5,
    "accusation": 1.0,
    "persuasion or seduction": 0.8,
    "no manipulation": 0.5  # downsample slightly
}


# -----------------------------
# AUGMENT FUNCTION
# -----------------------------
def augment_dataset(df):
    augmented_rows = []

    for _, row in df.iterrows():
        text = row["Dialogue"]
        labels = row["label"]

        label_names = df.columns[df.columns.str.contains("label")][0] if False else None

        # decode label indices if needed externally
        # here assume labels already decoded externally if multi-hot list exists

        # determine if we augment this row
        if random.random() < 0.5:
            aug_text = back_translate(text)

            if is_valid_aug(text, aug_text):
                new_row = row.copy()
                new_row["Dialogue"] = aug_text
                augmented_rows.append(new_row)

    return pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)