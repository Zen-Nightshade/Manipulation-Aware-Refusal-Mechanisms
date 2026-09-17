from pathlib import Path
import pandas as pd
import re
import numpy as np
from .config.paths import MAJ_P, MAJ_R, CON_R, CON_P


def clean_text(text: str):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('...', ',')
    return text


def preprocess(load_path: Path, save_path: Path):

    df = pd.read_csv(load_path)

    df = df.drop(columns=["ID", "Vulnerability"])

    # Normalize Technique column
    df["Technique"] = df["Technique"].replace(r'^\s*$', pd.NA, regex=True)
    df["Technique"] = df["Technique"].apply(
        lambda x: x.lower().strip() if isinstance(x, str) else x
    )

    df["Dialogue"] = df["Dialogue"].apply(clean_text)

    # -----------------------------
    # LABEL MAP (INCLUDING NO MANIPULATION)
    # -----------------------------
    technique_map = {
        "denial": 0,
        "evasion": 1,
        "feigning innocence": 2,
        "rationalization": 3,
        "playing victim role": 4,
        "playing servant role": 5,
        "shaming or belittlement": 6,
        "intimidation": 7,
        "brandishing anger": 8,
        "accusation": 9,
        "persuasion or seduction": 10,
        "no manipulation": 11
    }

    label_names = list(technique_map.keys())
    NUM_LABELS = len(label_names)

    # -----------------------------
    # DEBUG UNKNOWN LABELS
    # -----------------------------
    raw_labels = (
        df["Technique"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
    )

    unknown_labels = (set(raw_labels.unique()) | {"no manipulation"}) - set(technique_map.keys())

    # remove "no manipulation" from check since it's not in raw data
    unknown_labels = set(raw_labels.unique()) - set(technique_map.keys())

    if unknown_labels:
        print("UNKNOWN LABELS FOUND:")
        for u in sorted(unknown_labels):
            print(u)
        raise ValueError("Fix your dataset before proceeding.")

    # -----------------------------
    # ENCODING
    # -----------------------------
    def encode_row(row):
        vector = np.zeros(NUM_LABELS, dtype=int)

        manip = row["Manipulative"]
        tech = row["Technique"]

        # Case 1: NOT MANIPULATIVE
        if manip == 0:
            vector[technique_map["no manipulation"]] = 1
            return vector

        # Case 2: MANIPULATIVE → must have techniques
        if pd.isna(tech):
            raise ValueError("Manipulative=1 but Technique is missing")

        techniques = [t.strip() for t in tech.split(",")]

        for t in techniques:
            vector[technique_map[t]] = 1

        return vector
    # Remove inconsistent rows: Manipulative=1 but no Technique
    invalid_mask = (df["Manipulative"] == 1) & (df["Technique"].isna())

    print(f"Removing {invalid_mask.sum()} invalid rows (Manipulative=1 but no Technique)")

    df = df[~invalid_mask].reset_index(drop=True)

    df["label"] = df.apply(encode_row, axis=1)

    labels = np.vstack(df["label"].values)

    # -----------------------------
    # SANITY CHECK (CRITICAL)
    # -----------------------------
    no_manip_idx = technique_map["no manipulation"]

    for i, vec in enumerate(labels):
        if vec[no_manip_idx] == 1 and vec.sum() > 1:
            raise ValueError(f"Row {i} has 'no manipulation' + other labels")

    # -----------------------------
    # LABEL DISTRIBUTION
    # -----------------------------
    pos_counts = labels.sum(axis=0)

    print("\n=== LABEL DISTRIBUTION ===")
    for name, count in zip(label_names, pos_counts):
        print(f"{name}: {count}")

    # -----------------------------
    # OPTIONAL: REMOVE ZERO LABELS
    # -----------------------------
    valid_classes = pos_counts > 0

    if not valid_classes.all():
        print("\nRemoving zero-frequency labels:")
        for name, keep in zip(label_names, valid_classes):
            if not keep:
                print(name)

        labels = labels[:, valid_classes]
        label_names = [n for n, k in zip(label_names, valid_classes) if k]

    df["label"] = labels.tolist()
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    preprocess(
        load_path=CON_R,
        save_path=CON_P
    )