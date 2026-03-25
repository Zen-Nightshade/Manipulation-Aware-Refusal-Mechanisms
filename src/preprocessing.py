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
    df =df.drop(columns= ["ID", "Vulnerability"])
    df =df.dropna()

    df["Dialogue"] = df["Dialogue"].apply(clean_text)

    # print(df.info())
    # print(df.describe())
    # print(df["Dialogue"].iloc[2])

    df.to_csv(save_path, index= False)

if __name__ == "__main__":
    preprocess(
        load_path= MAJ_R,
        save_path= MAJ_P
    )