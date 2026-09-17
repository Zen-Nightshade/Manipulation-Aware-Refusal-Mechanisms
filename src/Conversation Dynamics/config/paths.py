from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

# Raw datasets
CON_R = RAW_DIR / "mentalmanip_con.csv"
DETAILED_R = RAW_DIR / "mentalmanip_detailed.csv"
MAJ_R = RAW_DIR / "mentalmanip_maj.csv"

# Preprocessed datasets
CON_P = PREPROCESSED_DIR / "mentalmanip_con.csv"
DETAILED_P = PREPROCESSED_DIR / "mentalmanip_detailed.csv"
MAJ_P = PREPROCESSED_DIR / "mentalmanip_maj.csv"

# Augmented datasets
CON_A = PREPROCESSED_DIR / "mentalmanip_con_aug.csv"
MAJ_A = PREPROCESSED_DIR / "mentalmanip_maj_aug.csv"

# Conversation Dynamics specific paths
CD_DIR = BASE_DIR / "src" / "Conversation Dynamics"
EMBEDDINGS_CACHE_PATH = CD_DIR / "embeddings_cache.pkl"
