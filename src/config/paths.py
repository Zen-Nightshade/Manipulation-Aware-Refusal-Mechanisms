from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

CON_R = RAW_DIR / "mentalmanip_con.csv"
DETAILED_R = RAW_DIR / "mentalmanip_detailed.csv"
MAJ_R = RAW_DIR / "mentalmanip_maj.csv"

CON_P = PREPROCESSED_DIR / "mentalmanip_con.csv"
DETAILED_P = PREPROCESSED_DIR / "mentalmanip_detailed.csv"
MAJ_P = PREPROCESSED_DIR / "mentalmanip_maj.csv"