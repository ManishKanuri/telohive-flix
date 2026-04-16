import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils import build_item_text

DATA_PATH = os.getenv("DATA_PATH", "/app/data/netflix_data.csv")
CACHE_PATH = os.getenv("CACHE_PATH", "/app/cache/embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH).fillna("")
    texts = df.apply(build_item_text, axis=1).tolist()
    print(f"  {len(texts)} titles loaded.")

    print(f"Loading model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(texts)} items ...")
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.save(CACHE_PATH, embeddings)
    print(f"Saved embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
    print(f"  → {CACHE_PATH}")


if __name__ == "__main__":
    main()
    sys.exit(0)
