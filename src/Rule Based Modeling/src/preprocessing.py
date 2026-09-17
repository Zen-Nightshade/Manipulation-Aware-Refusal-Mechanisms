from pathlib import Path
import pandas as pd
import re
from .config.paths import CON_P, CON_R, DETAILED_P, DETAILED_R, MAJ_P, MAJ_R

def clean_text(text: str):
    text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('...', ',')
    return text


# mentalmanip_con
def preprocess(load_path: Path, save_path: Path):

    df = pd.read_csv(load_path)

    df = df.drop(columns=["ID", "Vulnerability"])

    df["Technique"] = df["Technique"].replace(r'^\s*$', pd.NA, regex=True)
    df["Technique"] = df["Technique"].apply(lambda x: x.lower() if isinstance(x, str) else x)

    # Apply rule: if Manipulative == 1, Technique must exist
    df = df[~((df["Manipulative"] == 1) & (df["Technique"].isna()))]

    df["Dialogue"] = df["Dialogue"].apply(clean_text)

    df.to_csv(save_path, index=False)

    # print(df.info())
    # print(df.describe())
    # print(df["Dialogue"].iloc[2])

if __name__ == "__main__":
    preprocess(
        load_path= CON_R,
        save_path= CON_P
    )