import json

import pytest

from src.llm_provider import OpenAIProvider
from src.recommender import BookRecommender


def test_missing_api_key_marks_provider_unavailable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider()
    assert provider.available is False


def test_extract_preferences_validates_llm_json(monkeypatch):
    class FakeChoices:
        def __init__(self, content):
            self.message = type("Message", (), {"content": content})()

    class FakeResponse:
        def __init__(self, content):
            self.choices = [FakeChoices(content)]

    class FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": type("Completions", (), {"create": lambda self, **kwargs: FakeResponse(json.dumps({
                "genres": ["fantasy"],
                "topics": ["magic and adventure"],
                "authors": ["J.R.R. Tolkien"],
                "reading_level": "beginner",
                "tone": "adventurous",
                "goals": ["fun"],
                "exclusions": [],
                "count": 3,
            }))})()})()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("src.llm_provider.OpenAI", lambda *args, **kwargs: FakeClient())
    provider = OpenAIProvider()
    prefs = provider.extract_preferences("I want a beginner-friendly fantasy adventure")
    assert prefs["genres"] == ["fantasy"]
    assert prefs["count"] == 3


def test_hybrid_recommendations_fallback_without_llm():
    engine = BookRecommender()
    results = engine.hybrid_recommendations("I want a fantasy adventure book", limit=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    assert len(results) > 0
