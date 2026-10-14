from __future__ import annotations

import json
from pathlib import Path

from src.recommender import BookRecommender

APP_ROOT = Path(__file__).resolve().parent.parent
INTENTS_PATH = APP_ROOT / "data" / "raw" / "intents.json"


class BookBot:
    def __init__(
        self,
        intents_path: str | Path | None = None,
        recommender: BookRecommender | None = None,
    ):
        path = Path(intents_path) if intents_path else INTENTS_PATH
        if not path.exists():
            self.intents = []
        else:
            try:
                with path.open("r", encoding="utf-8") as file:
                    payload = json.load(file)
                self.intents = payload.get("intents", []) if isinstance(payload, dict) else []
            except (json.JSONDecodeError, OSError):
                self.intents = []
        self.recommender = recommender or BookRecommender()

    def _match_tag(self, message: str) -> str:
        normalized = " ".join(str(message).strip().lower().split())
        if not normalized:
            return "unknown"

        for intent in self.intents:
            if not isinstance(intent, dict):
                continue
            for pattern in intent.get("patterns", []):
                if isinstance(pattern, str) and pattern.lower() in normalized:
                    return intent.get("tag", "unknown")

        if any(token in normalized for token in ["hello", "hi", "hey", "greetings", "good morning", "good evening"]):
            return "greeting"
        if any(token in normalized for token in ["bye", "goodbye", "farewell", "see you", "talk later"]):
            return "goodbye"
        if any(token in normalized for token in ["thanks", "thank you", "appreciate", "grateful"]):
            return "thanks"
        if any(token in normalized for token in ["fantasy", "mystery", "romance", "science fiction", "dystopian", "classic", "genre", "books in"]):
            return "genre_request"
        if any(token in normalized for token in ["book", "recommend", "find", "search", "author"]):
            return "book_search"
        return "unknown"

    def respond(self, message: str) -> str:
        normalized = " ".join(str(message).strip().split())
        if not normalized:
            return "I’m not sure what you want. Try asking for a title, author, or genre."

        tag = self._match_tag(normalized)
        intent = next((item for item in self.intents if item.get("tag") == tag), None)

        if tag == "greeting":
            choices = intent.get("responses", []) if intent else []
            return choices[0] if choices else "Hello! I can recommend books by title, author, or genre."
        if tag == "goodbye":
            choices = intent.get("responses", []) if intent else []
            return choices[0] if choices else "Goodbye! Come back when you want your next read."
        if tag == "thanks":
            choices = intent.get("responses", []) if intent else []
            return choices[0] if choices else "You’re welcome! Happy reading."

        if tag == "genre_request":
            genre_words = ["fantasy", "mystery", "romance", "science fiction", "dystopian", "classic"]
            genre = next((word for word in genre_words if word in normalized.lower()), "fiction")
            books = self.recommender.category_recommendations(genre, limit=3)
            if books:
                titles = ", ".join(book.get("title", "Unknown") for book in books[:3])
                return f"Here are some {genre} picks: {titles}."
            return "I couldn’t find enough matches in that genre, but I can suggest popular books by mood or author."

        if tag == "book_search":
            books = self.recommender.search_books(normalized, limit=3)
            if books:
                titles = ", ".join(book["title"] for book in books[:3])
                return f"I found these matches: {titles}."
            return "I didn’t find a close match in the catalog. Try a different title, author, or keyword."

        return "I’m not sure I understood that. Try asking for a genre like fantasy, a title like The Hobbit, or a keyword like mystery."
