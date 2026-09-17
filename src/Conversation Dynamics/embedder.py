import csv
import pickle
import re
from sentence_transformers import SentenceTransformer
from config.paths import CON_R, EMBEDDINGS_CACHE_PATH

TECHNIQUE_MAP = {
    "denial": 0, "evasion": 1, "feigning innocence": 2, "rationalization": 3,
    "playing victim role": 4, "playing servant role": 5, "shaming or belittlement": 6,
    "intimidation": 7, "brandishing anger": 8, "accusation": 9,
    "persuasion or seduction": 10, "no manipulation": 11,
}


def parse_turns(dialogue):
    splits = re.split(r'(Person\d+):', str(dialogue))
    return [
        (splits[i], splits[i + 1].strip())
        for i in range(1, len(splits), 2)
    ]


def build_label(row):
    label = [0] * len(TECHNIQUE_MAP)
    if int(row.get("Manipulative", 0)) == 0:
        label[TECHNIQUE_MAP["no manipulation"]] = 1
    else:
        for t in row.get("Technique", "").split(","):
            t = t.strip().lower()
            if t in TECHNIQUE_MAP:
                label[TECHNIQUE_MAP[t]] = 1
    return label


def main():
    if not CON_R.exists():
        print(f"Dataset not found at {CON_R}")
        return

    with open(CON_R, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows.")

    unique_texts = list({text for row in rows for _, text in parse_turns(row["Dialogue"])})
    print(f"Encoding {len(unique_texts)} unique utterances...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    text2emb = dict(zip(unique_texts, model.encode(unique_texts, show_progress_bar=True, batch_size=64)))

    cached = []
    for row in rows:
        turns = parse_turns(row["Dialogue"])
        cached.append({
            "id": row.get("ID", ""),
            "turns": [{"speaker": s, "text": t, "embedding": text2emb[t]} for s, t in turns],
            "label": build_label(row),
        })

    with open(EMBEDDINGS_CACHE_PATH, "wb") as f:
        pickle.dump(cached, f)
    print(f"Cached to {EMBEDDINGS_CACHE_PATH}")


if __name__ == "__main__":
    main()
