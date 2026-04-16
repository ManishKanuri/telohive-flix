import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from anthropic import AsyncAnthropic
from sentence_transformers import SentenceTransformer

from utils import build_item_text

logger = logging.getLogger(__name__)

DATA_PATH    = os.getenv("DATA_PATH",  "/app/data/netflix_data.csv")
CACHE_PATH   = os.getenv("CACHE_PATH", "/app/cache/embeddings.npy")
MODEL_NAME   = "all-MiniLM-L6-v2"
LLM_MODEL    = "claude-haiku-4-5-20251001"

_EXPAND_PROMPT = """\
You are a semantic search expert for a Netflix content catalogue.

User query: "{query}"

Expand this into rich search keywords that will surface the best matches.

Think about:
- Core genres (thriller, drama, comedy, horror, romance, action, documentary, sci-fi, fantasy, crime, mystery, animation)
- Mood and tone (dark, uplifting, tense, funny, emotional, scary, gritty, heartwarming, satirical, surreal)
- Narrative devices (unreliable narrator, plot twist, heist, revenge arc, redemption, coming-of-age, time loop)
- Setting and era (1980s nostalgia, period drama, dystopian future, cold war, post-apocalyptic, medieval)
- Origin / language cues (Korean, Spanish, British, French, Japanese, Scandinavian, Bollywood)
- Audience signals (family-friendly, adult, teen drama, kids)
- Thematic elements (addiction, grief, identity, class struggle, survival, first love, found family)
- Content type if implied (movie / film → Movie; series / show / episodes / seasons → TV Show)

Output ONLY a single line of space-separated keywords. No punctuation beyond spaces. No explanation."""

_MOVIE_SIGNALS = {"movie", "film", "cinema", "feature"}
_SHOW_SIGNALS  = {"show", "series", "season", "seasons", "episode", "episodes",
                  "binge", "sitcom", "anime", "miniseries", "docuseries"}


def _detect_type_pref(query: str) -> str | None:
    tokens = set(query.lower().split())
    m = len(tokens & _MOVIE_SIGNALS)
    s = len(tokens & _SHOW_SIGNALS)
    if m > s:
        return "Movie"
    if s > m:
        return "TV Show"
    return None


class Recommender:
    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None
        self._embeddings: np.ndarray | None = None
        self._model: SentenceTransformer | None = None
        self._client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        self._ready = False

    def load(self) -> None:
        logger.info("Loading data …")
        self._df = pd.read_csv(DATA_PATH).fillna("")
        logger.info("  %d titles loaded.", len(self._df))

        logger.info("Loading embedding model …")
        self._model = SentenceTransformer(MODEL_NAME)

        logger.info("Loading embeddings from %s …", CACHE_PATH)
        self._embeddings = np.load(CACHE_PATH)
        logger.info("  Shape: %s", self._embeddings.shape)

        self._ready = True
        logger.info("Ready.")

    async def _expand_query(self, query: str) -> str:
        if not os.getenv("ANTHROPIC_API_KEY", ""):
            return query
        try:
            response = await self._client.messages.create(
                model=LLM_MODEL,
                max_tokens=160,
                messages=[{"role": "user", "content": _EXPAND_PROMPT.format(query=query)}],
            )
            expansion = response.content[0].text.strip()
            logger.info("Query expanded: %r → %r", query, expansion)
            return f"{query} {expansion}"
        except Exception as exc:
            logger.warning("Query expansion failed (%s) — using raw query.", exc)
            return query

    async def recommend(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not self._ready:
            raise RuntimeError("Recommender not initialised.")

        expanded = await self._expand_query(query)

        query_vec = self._model.encode(
            [expanded],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = (self._embeddings @ query_vec.T).flatten()

        type_pref = _detect_type_pref(query)
        if type_pref:
            match_mask = (self._df["type"] == type_pref).values.astype(float)
            scores = scores * (1.0 + 0.15 * match_mask)
            logger.info("Applied %s boost.", type_pref)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            row  = self._df.iloc[int(idx)]
            desc = str(row.get("description", "") or "")
            cast_raw = str(row.get("cast", "") or "")
            cast_display = ", ".join(c.strip() for c in cast_raw.split(",")[:6] if c.strip())
            results.append({
                "title":       str(row.get("title",        "")),
                "type":        str(row.get("type",         "")),
                "year":        str(row.get("release_year", "")),
                "rating":      str(row.get("rating",       "")),
                "duration":    str(row.get("duration",     "")),
                "genres":      str(row.get("listed_in",    "")),
                "description": desc[:300] + "…" if len(desc) > 300 else desc,
                "full_description": desc,
                "director":    str(row.get("director",     "")),
                "cast":        cast_display,
                "country":     str(row.get("country",      "")),
                "score":       round(float(scores[idx]), 4),
            })

        return results
