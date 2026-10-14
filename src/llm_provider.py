from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

try:  # pragma: no cover - optional dependency in test/mocked environments
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

load_dotenv()

logger = logging.getLogger("bookwise.llm")


class LLMProvider:
    """Minimal provider interface for optional LLM-backed recommendation enrichment."""

    available: bool = False

    def extract_preferences(self, query: str) -> dict[str, Any]:
        return {}

    def generate_reasoning(self, query: str, candidates: list[dict[str, Any]]) -> str:
        return ""


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self.model = (model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        self.timeout = timeout
        self.client = None
        self.available = bool(self.api_key)

        if not self.available:
            logger.info("OPENAI_API_KEY is not configured; LLM features remain unavailable.")
            return

        try:
            if OpenAI is None:
                raise RuntimeError("The OpenAI SDK is not installed.")
            self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        except Exception as exc:  # pragma: no cover - depends on runtime SDK
            logger.warning("Unable to initialize OpenAI client: %s", exc)
            self.available = False
            self.client = None

    @staticmethod
    def _default_preferences() -> dict[str, Any]:
        return {
            "genres": [],
            "topics": [],
            "authors": [],
            "reading_level": "any",
            "tone": "any",
            "goals": [],
            "exclusions": [],
            "count": 5,
        }

    def extract_preferences(self, query: str) -> dict[str, Any]:
        if not self.available or not self.client:
            return self._default_preferences()

        normalized = " ".join((query or "").strip().split())
        if not normalized:
            return self._default_preferences()

        prompt = (
            "Extract structured reading preferences from the user's request. "
            "Return valid JSON with keys: genres, topics, authors, reading_level, tone, goals, exclusions, count. "
            "Use plain text values and arrays; if information is missing, use empty arrays or 'any'."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": normalized},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            parsed = json.loads(content) if content else {}
        except Exception as exc:  # pragma: no cover - network/error paths
            logger.warning("LLM preference extraction failed: %s", exc)
            return self._default_preferences()

        preferences = self._default_preferences()
        if not isinstance(parsed, dict):
            return preferences

        for key in preferences:
            value = parsed.get(key, preferences[key])
            if key == "count":
                try:
                    preferences[key] = max(1, min(int(value), 10))
                except (TypeError, ValueError):
                    preferences[key] = 5
            elif isinstance(value, list):
                preferences[key] = [str(item).strip() for item in value if str(item).strip()]
            elif isinstance(value, str):
                preferences[key] = value.strip() if key in {"reading_level", "tone"} else value
            else:
                preferences[key] = preferences[key]

        return preferences

    def generate_reasoning(self, query: str, candidates: list[dict[str, Any]]) -> str:
        if not self.available or not self.client or not candidates:
            return ""

        trimmed_candidates = []
        for candidate in candidates[:6]:
            title = candidate.get("title") or "Untitled"
            authors = candidate.get("authors") or "Unknown author"
            description = (candidate.get("description") or "").strip()
            trimmed_candidates.append({
                "title": title,
                "authors": authors,
                "categories": candidate.get("categories") or "General",
                "description": description[:220],
            })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a librarian. Explain why the candidate books fit the user's reading request using only the provided catalog metadata. Return valid JSON with a single 'summary' string field.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps({
                            "query": (query or "").strip(),
                            "candidate_books": trimmed_candidates,
                        }, ensure_ascii=False),
                    },
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            parsed = json.loads(content) if content else {}
            return str(parsed.get("summary") or "").strip()
        except Exception as exc:  # pragma: no cover - network/error paths
            logger.warning("LLM reasoning generation failed: %s", exc)
            return ""
