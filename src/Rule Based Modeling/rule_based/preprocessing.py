import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer


def prepare_data(path):
    df = pd.read_csv(path)

    # Convert labels to list
    df['labels_list'] = df['Technique'].apply(
        lambda x: [t.strip() for t in x.split(',')] if pd.notna(x) else []
    )

    # Encode labels
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df['labels_list'])

    # Text
    X_text = df['Dialogue'].values

    return X_text, Y, mlb


def split_speakers(dialogue: str):
    """
    Splits dialogue into Person1 and Person2 text.
    Returns: (person1_text, person2_text)
    """

    # Find all speaker-text pairs
    pattern = r"(Person1|Person2):\s*(.*?)(?=(Person1|Person2):|$)"
    matches = re.findall(pattern, dialogue, flags=re.DOTALL)

    p1_lines = []
    p2_lines = []

    for speaker, text, _ in matches:
        text = text.strip()

        if speaker == "Person1":
            p1_lines.append(text)
        else:
            p2_lines.append(text)

    person1_text = " ".join(p1_lines)
    person2_text = " ".join(p2_lines)

    return person1_text, person2_text