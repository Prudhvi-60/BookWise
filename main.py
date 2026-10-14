from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.chatbot import BookBot
from src.data_processing import get_dataset_stats, prepare_dataset
from src.llm_provider import OpenAIProvider
from src.recommender import BookRecommender

load_dotenv()

APP_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_ROOT / "templates"
STATIC_DIR = APP_ROOT / "static"

logger = logging.getLogger("bookwise")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

processed_df = prepare_dataset()

llm_provider = OpenAIProvider()
recommender = BookRecommender(llm_provider=llm_provider, dataset=processed_df)
chatbot = BookBot(recommender=recommender)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("BookWise started with %s books.", len(recommender.df))
    if llm_provider.available:
        logger.info("LLM recommendations are enabled via %s.", llm_provider.model)
    else:
        logger.info("LLM recommendations are disabled; using catalog-based recommendation logic.")
    yield


app = FastAPI(title="BookWise — Discover Your Next Read", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatMessage(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    stats = get_dataset_stats(recommender.df)
    return templates.TemplateResponse(request=request, name="index.html", context={"stats": stats})


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> dict:
    ready = not recommender.df.empty and recommender.matrix.shape[0] == len(recommender.df)
    return {
        "status": "ok" if ready else "unavailable",
        "books": int(len(recommender.df)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "llm_available": bool(llm_provider.available),
    }


@app.get("/api/stats")
def api_stats() -> dict:
    return get_dataset_stats(recommender.df)


@app.get("/api/categories")
def api_categories() -> dict:
    categories = sorted({
        category.strip().lower()
        for row in recommender.df["categories"].fillna("").astype(str).tolist()
        for category in row.split(",") if category.strip()
    })
    return {"categories": categories}


@app.get("/api/search")
def api_search(q: str = Query(..., min_length=1, max_length=200)) -> dict:
    books = recommender.search_books(q, limit=12)
    return {"query": q, "count": len(books), "books": books}


@app.get("/api/recommendations")
def api_recommendations(
    title: str | None = None,
    q: str | None = None,
    genre: str | None = None,
    limit: int = Query(default=6, gt=0, le=20),
) -> dict:
    query_text = (title or q or genre or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Please provide a title, query, or genre.")
    if genre:
        books = recommender.category_recommendations(genre, limit=limit)
    elif title:
        books = recommender.recommend_for_book(title, limit=limit)
    else:
        books = recommender.hybrid_recommendations(query_text, limit=limit, llm_provider=llm_provider)
    llm_assisted = bool(llm_provider.available and not genre and not title)
    return {"query": query_text, "count": len(books), "books": books, "llm_assisted": llm_assisted}


@app.post("/api/chat")
def api_chat(payload: ChatMessage) -> dict:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="A message is required.")

    response = chatbot.respond(message)
    books: list[dict] = []
    if any(word in message.lower() for word in ["recommend", "suggest", "want", "interested", "like", "need", "for someone"]):
        books = recommender.hybrid_recommendations(message, limit=3, llm_provider=llm_provider)
        response = response if not books else response
    elif "fantasy" in message.lower() or "mystery" in message.lower() or "romance" in message.lower():
        books = recommender.category_recommendations(next((word for word in ["fantasy", "mystery", "romance", "science fiction"] if word in message.lower()), "fiction"), limit=3)
    else:
        books = recommender.search_books(message, limit=3)
    return {"response": response, "books": books, "llm_assisted": bool(llm_provider.available)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
