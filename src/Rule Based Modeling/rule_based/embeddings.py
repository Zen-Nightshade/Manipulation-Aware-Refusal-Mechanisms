import numpy as np
import os
import pickle
from typing import Tuple, Optional

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
except:
    GPU_AVAILABLE = False
    DEVICE = "cpu"

print(f"[INFO] GPU Available: {GPU_AVAILABLE} | Device: {DEVICE}")


def get_sentence_transformer_embeddings(texts: np.ndarray, model_name: str = 'all-MiniLM-L6-v2', cache_path: Optional[str] = None) -> np.ndarray:
    """
    Extract embeddings using SentenceTransformers (fastest, lightweight)
    
    Args:
        texts: Array of text strings
        model_name: SentenceTransformer model name
        cache_path: Path to save/load embeddings
    
    Returns:
        Embeddings array (n_samples, embedding_dim)
    """
    from sentence_transformers import SentenceTransformer
    
    # Check if cached
    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"[INFO] Extracting SentenceTransformer embeddings ({model_name})...")
    model = SentenceTransformer(model_name)
    
    # Use GPU if available
    if GPU_AVAILABLE:
        model = model.to(DEVICE)
    
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Cache embeddings
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"[INFO] Cached embeddings to {cache_path}")
    
    return embeddings


def get_distilbert_embeddings(texts: np.ndarray, cache_path: Optional[str] = None) -> np.ndarray:
    """
    Extract embeddings using DistilBERT (lightweight BERT)
    
    Args:
        texts: Array of text strings
        cache_path: Path to save/load embeddings
    
    Returns:
        Embeddings array (n_samples, 768)
    """
    from transformers import DistilBertTokenizer, DistilBertModel
    import torch
    
    # Check if cached
    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"[INFO] Extracting DistilBERT embeddings...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    if GPU_AVAILABLE:
        model = model.to(DEVICE)
    
    embeddings = []
    batch_size = 32
    
    print(f"[INFO] Processing {len(texts)} texts in batches of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(list(batch), return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        if GPU_AVAILABLE:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token (first token) as sequence embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)
    
    embeddings = np.vstack(embeddings)
    
    # Cache embeddings
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"[INFO] Cached embeddings to {cache_path}")
    
    return embeddings


def get_roberta_embeddings(texts: np.ndarray, cache_path: Optional[str] = None) -> np.ndarray:
    """
    Extract embeddings using RoBERTa (contextual, better than BERT)
    
    Args:
        texts: Array of text strings
        cache_path: Path to save/load embeddings
    
    Returns:
        Embeddings array (n_samples, 768)
    """
    from transformers import RobertaTokenizer, RobertaModel
    import torch
    
    # Check if cached
    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"[INFO] Extracting RoBERTa embeddings...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    
    if GPU_AVAILABLE:
        model = model.to(DEVICE)
    
    embeddings = []
    batch_size = 32
    
    print(f"[INFO] Processing {len(texts)} texts in batches of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(list(batch), return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        if GPU_AVAILABLE:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token as sequence embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)
    
    embeddings = np.vstack(embeddings)
    
    # Cache embeddings
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"[INFO] Cached embeddings to {cache_path}")
    
    return embeddings


def get_embeddings(texts: np.ndarray, transformer_name: str = 'sentence-transformers', cache_dir: Optional[str] = None) -> np.ndarray:
    """
    Unified interface to get embeddings from different transformers
    
    Args:
        texts: Array of text strings
        transformer_name: One of ['sentence-transformers', 'distilbert', 'roberta']
        cache_dir: Directory to cache embeddings
    
    Returns:
        Embeddings array
    """
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{transformer_name}_embeddings.pkl")
    
    if transformer_name == 'sentence-transformers':
        return get_sentence_transformer_embeddings(texts, cache_path=cache_path)
    elif transformer_name == 'distilbert':
        return get_distilbert_embeddings(texts, cache_path=cache_path)
    elif transformer_name == 'roberta':
        return get_roberta_embeddings(texts, cache_path=cache_path)
    else:
        raise ValueError(f"Unknown transformer: {transformer_name}")


def extract_speaker_embeddings(p1_embedding: np.ndarray, p2_embedding: np.ndarray) -> np.ndarray:
    """
    Extract speaker-specific features from embeddings
    
    Args:
        p1_embedding: Person1 embedding
        p2_embedding: Person2 embedding
    
    Returns:
        Combined features including speaker embeddings and interaction features
    """
    from scipy.spatial.distance import cosine
    
    # Cosine similarity between speakers (dominance indicator)
    speaker_distance = cosine(p1_embedding, p2_embedding) if np.linalg.norm(p1_embedding) > 0 and np.linalg.norm(p2_embedding) > 0 else 0.0
    
    # Magnitude difference
    magnitude_diff = np.linalg.norm(p1_embedding) - np.linalg.norm(p2_embedding)
    
    # Concatenate: p1_emb, p2_emb, distance, magnitude_diff
    features = np.concatenate([
        p1_embedding,
        p2_embedding,
        [speaker_distance, magnitude_diff]
    ])
    
    return features
