import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recommender import Recommender

STATIC_DIR = os.getenv("STATIC_DIR", "/app/static")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

_recommender = Recommender()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _recommender.load()
    yield


app = FastAPI(title="Movie Recommender", lifespan=lifespan)


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)


@app.get("/api/health")
async def health():
    return {"status": "ok", "ready": _recommender._ready}


@app.post("/api/recommend")
async def recommend(req: RecommendRequest):
    try:
        results = await _recommender.recommend(req.query.strip(), req.top_k)
        return JSONResponse({"query": req.query, "results": results})
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Recommendation error")
        raise HTTPException(status_code=500, detail="Internal server error")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
