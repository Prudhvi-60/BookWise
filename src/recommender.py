from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_processing import clean_dataset, get_dataset_path, load_dataset
from src.llm_provider import OpenAIProvider


class BookRecommender:
    GENRE_ALIASES = {
        "Fantasy": ["fantasy", "magic", "adventure"],
        "Science Fiction": ["science fiction", "sci-fi", "sf", "dystopian", "future"],
        "Mystery": ["mystery", "detective", "crime"],
        "Thriller": ["thriller", "suspense", "adrenaline"],
        "Romance": ["romance", "love story", "relationship"],
        "Historical Fiction": ["historical fiction", "historical", "classic"],
        "Horror": ["horror", "dark", "haunting"],
        "Biography": ["biography", "memoir"],
        "Self-Help": ["self-help", "self help", "personal growth"],
        "Philosophy": ["philosophy", "thinker", "thought-provoking"],
        "Psychology": ["psychology", "mind", "behavior"],
        "Business": ["business", "strategy", "leadership"],
        "Technology": ["technology", "computers", "innovation"],
        "Young Adult": ["young adult", "ya"],
        "Adventure": ["adventure"],
    }

    def __init__(
        self,
        dataset_path: str | None = None,
        llm_provider: OpenAIProvider | None = None,
        dataset: pd.DataFrame | None = None,
    ):
        df = dataset.copy() if dataset is not None else clean_dataset(load_dataset(dataset_path or get_dataset_path()))
        self.df = df.copy().reset_index(drop=True)
        self.df["search_text"] = (
            self.df[["title", "subtitle", "authors", "categories", "description"]].fillna("").astype(str).agg(lambda row: " ".join(row), axis=1)
        )
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.df["search_text"].fillna(""))
        self._title_lookup = {str(title).strip().lower(): idx for idx, title in enumerate(self.df["title"].fillna(""))}
        self._isbn_lookup = {str(isbn).strip(): idx for isbn, idx in zip(self.df["isbn13"].fillna(""), range(len(self.df)))}
        self.llm_provider = llm_provider
        self.global_rating_mean = float(self.df["average_rating"].mean()) if not self.df["average_rating"].dropna().empty else 0.0
        self.global_rating_count = float(self.df["ratings_count"].fillna(0).median()) if not self.df["ratings_count"].dropna().empty else 0.0

    @staticmethod
    def _normalize_query(query: str) -> str:
        if query is None:
            return ""
        return " ".join(str(query).strip().lower().split())

    @staticmethod
    def _safe_categories(row: pd.Series) -> set[str]:
        raw = str(row.get("categories", "") or "")
        return {part.strip().lower() for part in raw.split(",") if part and part.strip()}

    @staticmethod
    def _safe_authors(value: Any) -> list[str]:
        if value is None or pd.isna(value):
            return []
        return [part.strip() for part in str(value).split(",") if part and part.strip()]

    @staticmethod
    def _as_book_record(row: pd.Series) -> dict:
        def clean(value):
            if value is None or pd.isna(value):
                return None
            if isinstance(value, np.generic):
                return value.item()
            return value

        return {
            "isbn13": clean(row.get("isbn13")),
            "isbn10": clean(row.get("isbn10")),
            "title": str(clean(row.get("title")) or "").strip() or None,
            "subtitle": clean(row.get("subtitle")),
            "authors": clean(row.get("authors")),
            "categories": clean(row.get("categories")),
            "thumbnail": clean(row.get("thumbnail")),
            "description": clean(row.get("description")),
            "published_year": clean(row.get("published_year")),
            "average_rating": clean(row.get("average_rating")),
            "num_pages": clean(row.get("num_pages")),
            "ratings_count": clean(row.get("ratings_count")),
        }

    @staticmethod
    def _coerce_numeric(value: Any, default: float = 0.0) -> float:
        if value is None or pd.isna(value):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalized_tokens(value: Any) -> set[str]:
        text = " ".join(str(value or "").lower().split())
        return {token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2}

    @staticmethod
    def _category_matches(book_categories: Any, wanted: list[str]) -> int:
        if not wanted:
            return 0
        normalized = {token.strip().lower() for token in str(book_categories or "").split(",") if token.strip()}
        matches = 0
        for item in wanted:
            name = item.strip().lower()
            if name in normalized:
                matches += 1
            for category in normalized:
                if category.startswith(name) or name.startswith(category):
                    matches += 0.5
        return int(matches)

    def _find_book_index(self, title_or_isbn: str) -> int | None:
        normalized = self._normalize_query(title_or_isbn)
        if not normalized:
            return None
        if normalized in self._isbn_lookup:
            return self._isbn_lookup[normalized]
        if normalized in self._title_lookup:
            return self._title_lookup[normalized]
        for idx, title in enumerate(self.df["title"].fillna("")):
            if normalized in str(title).strip().lower():
                return idx
        return None

    def parse_natural_language_preferences(self, query: str, candidate_authors: list[str] | None = None) -> dict[str, Any]:
        normalized = self._normalize_query(query)
        if not normalized:
            return {
                "query": "",
                "genres": [],
                "authors": [],
                "exclude_genres": [],
                "exclude_authors": [],
                "year_min": None,
                "year_max": None,
                "min_rating": None,
                "min_reviews": None,
                "page_min": None,
                "page_max": None,
                "language": None,
                "strict": True,
                "mode": "best_match",
                "text": "",
            }

        preferred_genres: list[str] = []
        excluded_genres: list[str] = []
        for canonical, aliases in self.GENRE_ALIASES.items():
            for alias in aliases:
                if alias in normalized:
                    preferred_genres.append(canonical)
                    break
        for token in ("avoid", "exclude", "not", "without"):
            if token in normalized:
                pass
        for canonical, aliases in self.GENRE_ALIASES.items():
            for alias in aliases:
                pattern = rf"(?:avoid|exclude|without|not)\s+(?:{re.escape(alias)}|{re.escape(canonical.lower())})"
                if re.search(pattern, normalized):
                    excluded_genres.append(canonical)
                    break

        author_candidates = candidate_authors or [str(row.get("authors") or "") for _, row in self.df.iterrows()]
        author_matches: list[str] = []
        for name in sorted({author.strip() for author in author_candidates if author and author.strip()}, key=len, reverse=True):
            lowered = name.lower()
            if lowered in normalized or normalized.find(lowered) != -1:
                author_matches.append(name)
        normalized_authors = []
        if "by " in normalized:
            pieces = normalized.split("by ", 1)[1].split()
            if pieces:
                candidate = " ".join(pieces[:4])
                for author in author_candidates:
                    if author and candidate in author.lower():
                        normalized_authors.append(author)
        authors = sorted(set(author_matches + normalized_authors), key=lambda item: len(item))

        year_min = None
        year_max = None
        year_matches = re.findall(r"(?:after|since|from|published after|published since|from\s+)?\s*(\d{4})", normalized)
        if year_matches:
            year_min = max(int(year_matches[0]), 0)
            if "before" in normalized or "prior to" in normalized:
                year_max = max(int(year_matches[0]), 0)
        if "before " in normalized:
            match = re.search(r"before\s+(\d{4})", normalized)
            if match:
                year_max = int(match.group(1))
        if "after " in normalized:
            match = re.search(r"after\s+(\d{4})", normalized)
            if match:
                year_min = int(match.group(1))
        if "between " in normalized and " and " in normalized:
            match = re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", normalized)
            if match:
                year_min = int(match.group(1))
                year_max = int(match.group(2))
        if "recent" in normalized or "last five years" in normalized:
            year_min = 2020
        if "classic" in normalized and "modern" not in normalized:
            year_max = 1960

        min_rating = None
        match = re.search(r"(?:rating|rated|average)\s*(?:above|over|at least|>=|more than)?\s*(\d(?:\.\d)?)", normalized)
        if match:
            min_rating = float(match.group(1))
        if "highly rated" in normalized:
            min_rating = 4.0

        min_reviews = None
        match = re.search(r"(?:at least|minimum|more than|over)\s*(\d{1,3}(?:,\d{3})*)\s*(?:reviews|ratings)", normalized)
        if match:
            min_reviews = int(match.group(1).replace(",", ""))

        page_min = None
        page_max = None
        match = re.search(r"(?:under|less than|below|fewer than)\s*(\d{2,4})\s*(?:pages?|pp)", normalized)
        if match:
            page_max = int(match.group(1))
        match = re.search(r"(?:at least|minimum|over|more than)\s*(\d{2,4})\s*(?:pages?|pp)", normalized)
        if match:
            page_min = int(match.group(1))
        match = re.search(r"(\d{2,4})\s*(?:pages?|pp)", normalized)
        if match and page_min is None and page_max is None:
            page_max = int(match.group(1))

        language = None
        for candidate in ["english", "french", "spanish", "german", "italian"]:
            if candidate in normalized and "not " + candidate not in normalized:
                language = candidate
                break

        mode = "best_match"
        if "hidden gems" in normalized:
            mode = "hidden_gems"
        elif "popular" in normalized:
            mode = "popular"
        elif "discover" in normalized or "new" in normalized:
            mode = "discover"

        return {
            "query": normalized,
            "genres": preferred_genres,
            "authors": authors,
            "exclude_genres": excluded_genres,
            "exclude_authors": [],
            "year_min": year_min,
            "year_max": year_max,
            "min_rating": min_rating,
            "min_reviews": min_reviews,
            "page_min": page_min,
            "page_max": page_max,
            "language": language,
            "strict": True,
            "mode": mode,
            "text": query.strip(),
        }

    def _score_candidate_book(self, query: str, preference_terms: set[str], book: dict[str, Any]) -> float:
        title = (book.get("title") or "").lower()
        authors = (book.get("authors") or "").lower()
        categories = (book.get("categories") or "").lower()
        description = (book.get("description") or "").lower()
        text_blob = " ".join([title, authors, categories, description])

        score = 0.0
        query_tokens = {token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 2}
        for token in query_tokens | preference_terms:
            if token in text_blob:
                score += 0.18

        for candidate in (book.get("categories") or "").split(","):
            if candidate.strip().lower() in {term.lower() for term in preference_terms}:
                score += 0.3

        avg_rating = float(book.get("average_rating") or 0)
        score += (avg_rating / 5.0) * 0.2

        rating_count = float(book.get("ratings_count") or 0)
        score += min(rating_count / 10000000.0, 1.0) * 0.1

        if title in query.lower() or query.lower() in title:
            score += 0.4

        return score

    def _apply_preferences(self, frame: pd.DataFrame, preferences: dict[str, Any], strict: bool = True) -> pd.DataFrame:
        if frame.empty:
            return frame

        filtered = frame.copy()
        genres = [gen for gen in preferences.get("genres", []) if isinstance(gen, str) and gen.strip()]
        exclude_genres = [gen for gen in preferences.get("exclude_genres", []) if isinstance(gen, str) and gen.strip()]
        authors = [author for author in preferences.get("authors", []) if isinstance(author, str) and author.strip()]
        exclude_authors = [author for author in preferences.get("exclude_authors", []) if isinstance(author, str) and author.strip()]
        year_min = preferences.get("year_min")
        year_max = preferences.get("year_max")
        min_rating = preferences.get("min_rating")
        min_reviews = preferences.get("min_reviews")
        page_min = preferences.get("page_min")
        page_max = preferences.get("page_max")
        language = preferences.get("language")

        if exclude_genres:
            filtered = filtered[~filtered["categories"].fillna("").astype(str).str.lower().str.contains("|".join(re.escape(item.lower()) for item in exclude_genres), na=False, regex=True)]
        if exclude_authors:
            excluded_patterns = "|".join(re.escape(item.lower()) for item in exclude_authors)
            filtered = filtered[~filtered["authors"].fillna("").astype(str).str.lower().str.contains(excluded_patterns, na=False, regex=True)]

        if year_min is not None:
            filtered = filtered[filtered["published_year"].fillna(-1).astype(float) >= float(year_min)]
        if year_max is not None:
            filtered = filtered[filtered["published_year"].fillna(9999).astype(float) <= float(year_max)]
        if min_rating is not None:
            filtered = filtered[filtered["average_rating"].fillna(-1).astype(float) >= float(min_rating)]
        if min_reviews is not None:
            filtered = filtered[filtered["ratings_count"].fillna(0).astype(float) >= float(min_reviews)]
        if page_min is not None:
            filtered = filtered[filtered["num_pages"].fillna(99999).astype(float) >= float(page_min)]
        if page_max is not None:
            filtered = filtered[filtered["num_pages"].fillna(0).astype(float) <= float(page_max)]
        if language and "language" in filtered.columns:
            filtered = filtered[filtered["language"].fillna("").astype(str).str.lower().eq(language.lower())]

        if strict:
            for genre in genres:
                pattern = re.escape(genre.lower())
                filtered = filtered[filtered["categories"].fillna("").astype(str).str.lower().str.contains(pattern, na=False)]
            for author in authors:
                pattern = re.escape(author.lower())
                filtered = filtered[filtered["authors"].fillna("").astype(str).str.lower().str.contains(pattern, na=False)]

        return filtered.reset_index(drop=True)

    def recommend_for_preferences(self, preferences: dict[str, Any] | None = None, limit: int = 10, strict: bool = True) -> list[dict[str, Any]]:
        prefs = {
            "query": "",
            "genres": [],
            "authors": [],
            "exclude_genres": [],
            "exclude_authors": [],
            "year_min": None,
            "year_max": None,
            "min_rating": None,
            "min_reviews": None,
            "page_min": None,
            "page_max": None,
            "language": None,
            "strict": strict,
            "mode": "best_match",
            "text": "",
        }
        if isinstance(preferences, dict):
            prefs.update({key: value for key, value in preferences.items() if key in prefs})
            for key, value in preferences.items():
                if key not in prefs and key in {"description", "query_text", "text"}:
                    prefs["query"] = str(value)
                    prefs["text"] = str(value)

        query_terms = self._normalized_tokens(prefs.get("query") or prefs.get("text") or "")
        if prefs.get("genres"):
            query_terms |= {term.lower() for term in prefs["genres"]}
        if prefs.get("authors"):
            query_terms |= {term.lower() for term in prefs["authors"]}

        filtered = self._apply_preferences(self.df.copy(), prefs, strict=strict)
        if filtered.empty:
            return []

        scored_rows: list[dict[str, Any]] = []
        for _, row in filtered.iterrows():
            book = self._as_book_record(row)
            text_blob = " ".join([
                str(book.get("title") or ""),
                str(book.get("authors") or ""),
                str(book.get("categories") or ""),
                str(book.get("description") or ""),
            ]).lower()
            overlap = sum(1 for token in query_terms if token in text_blob)
            genre_count = self._category_matches(book.get("categories"), prefs.get("genres", []))
            author_bonus = 1 if any(author.lower() in str(book.get("authors") or "").lower() for author in prefs.get("authors", [])) else 0
            rating = self._coerce_numeric(book.get("average_rating"), 0.0)
            review_count = self._coerce_numeric(book.get("ratings_count"), 0.0)
            year = self._coerce_numeric(book.get("published_year"), 0.0)
            page_count = self._coerce_numeric(book.get("num_pages"), 0.0)
            year_min = prefs.get("year_min")
            year_max = prefs.get("year_max")
            min_rating = prefs.get("min_rating")
            page_min = prefs.get("page_min")
            page_max = prefs.get("page_max")

            score = 0.25 * min(overlap, 8) / 8.0
            score += 0.35 * min(genre_count, 3) / 3.0
            score += 0.2 * author_bonus
            score += 0.1 * min(rating / 5.0, 1.0)
            score += 0.1 * min(review_count / 1000000.0, 1.0)

            if year_min is not None and year and year >= float(year_min):
                score += 0.08
            if year_max is not None and year and year <= float(year_max):
                score += 0.08
            if page_min is not None and page_count and page_count >= float(page_min):
                score += 0.06
            if page_max is not None and page_count and page_count <= float(page_max):
                score += 0.06

            explanations: list[str] = []
            if genre_count:
                explanations.append("Matches your selected genres.")
            if author_bonus:
                explanations.append("Matches a preferred author.")
            if year_min is not None and year and year >= float(year_min):
                explanations.append("Falls within your publication-year range.")
            if page_max is not None and page_count and page_count <= float(page_max):
                explanations.append("Matches your maximum page-count preference.")
            if min_rating is not None and rating and rating >= float(min_rating):
                explanations.append("Meets your minimum rating threshold.")
            if overlap:
                explanations.append("Resembles your requested reading themes.")
            if not explanations:
                explanations = ["Selected based on catalogue relevance and quality signals."]

            scored_rows.append({
                **book,
                "match_score": round(min(score, 1.0), 4),
                "match_reasons": explanations[:3],
            })

        ranked = sorted(scored_rows, key=lambda item: (item["match_score"], item.get("average_rating") or 0, item.get("ratings_count") or 0), reverse=True)
        unique_books: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for item in ranked:
            title = str(item.get("title") or "").strip().lower()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            unique_books.append(item)
            if len(unique_books) >= limit:
                break
        return unique_books

    def search_books(self, query: str, limit: int = 10) -> list[dict]:
        normalized = self._normalize_query(query)
        if not normalized:
            return []

        literal_matches = self.df[
            self.df["title"].fillna("").str.lower().str.contains(normalized, na=False)
            | self.df["authors"].fillna("").str.lower().str.contains(normalized, na=False)
            | self.df["categories"].fillna("").str.lower().str.contains(normalized, na=False)
            | self.df["description"].fillna("").str.lower().str.contains(normalized, na=False)
        ].copy()

        if not literal_matches.empty:
            ranked = literal_matches.copy()
            ranked["match_score"] = 3.0
        else:
            query_vector = self.vectorizer.transform([normalized])
            cosine = cosine_similarity(query_vector, self.matrix).ravel()
            if float(cosine.max()) < 0.2:
                return []
            ranked_idx = np.argsort(cosine)[::-1][: max(limit * 5, 10)]
            ranked = self.df.iloc[ranked_idx].copy()
            ranked["match_score"] = cosine[ranked_idx]

        ranked = ranked.sort_values(
            by=["match_score", "average_rating", "ratings_count"],
            ascending=[False, False, False],
            na_position="last",
        )

        results = []
        seen_titles = set()
        for _, row in ranked.head(limit).iterrows():
            title = str(row.get("title") or "").strip()
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            record = self._as_book_record(row)
            record["match_reasons"] = ["Text similarity match for your search."]
            record["match_score"] = float(row.get("match_score", 0.0))
            results.append(record)
        return results

    def recommend_for_book(self, title_or_isbn: str, limit: int = 6) -> list[dict]:
        book_index = self._find_book_index(title_or_isbn)
        if book_index is None:
            return self.search_books(title_or_isbn, limit=limit)

        base_row = self.df.iloc[book_index]
        similarities = cosine_similarity(self.matrix[book_index], self.matrix).ravel()
        candidate_scores: list[tuple[int, float]] = []

        for idx, row in enumerate(self.df.itertuples(index=False)):
            if idx == book_index:
                continue
            score = float(similarities[idx])
            categories_base = self._safe_categories(base_row)
            categories_row = self._safe_categories(pd.Series(row._asdict()))
            category_overlap = len(categories_base & categories_row) / max(len(categories_base | categories_row), 1)
            rating = float(row.average_rating) if pd.notna(row.average_rating) else 0.0
            rating_boost = (rating / 5.0) * 0.15 if pd.notna(row.average_rating) else 0.0
            score = score + (0.08 * category_overlap) + rating_boost
            candidate_scores.append((idx, score))

        ranked = sorted(candidate_scores, key=lambda item: item[1], reverse=True)[: max(limit * 4, limit)]
        results = []
        seen_titles = set()
        for idx, _ in ranked:
            row = self.df.iloc[idx]
            title = str(row.get("title") or "").strip()
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            record = self._as_book_record(row)
            record["match_reasons"] = ["Similar to your selected title based on category and text similarity."]
            record["match_score"] = round(float(row.get("average_rating") or 0.0), 4)
            results.append(record)
            if len(results) >= limit:
                break

        if not results:
            return self.search_books(str(base_row["title"]), limit=limit)
        return results

    def recommend_for_query(self, query: str, limit: int = 6) -> list[dict]:
        normalized = self._normalize_query(query)
        if not normalized:
            return []
        return self.search_books(normalized, limit=limit)

    def category_recommendations(self, category: str, limit: int = 6) -> list[dict]:
        normalized = self._normalize_query(category)
        if not normalized:
            return []
        subset = self.df[
            self.df["categories"].fillna("").str.lower().str.contains(normalized, na=False)
            | self.df["title"].fillna("").str.lower().str.contains(normalized, na=False)
        ].copy()
        if subset.empty:
            return []
        subset = subset.sort_values(by=["average_rating", "ratings_count"], ascending=[False, False], na_position="last")
        results = []
        for _, row in subset.head(limit).iterrows():
            record = self._as_book_record(row)
            record["match_reasons"] = ["Matches the selected category."]
            record["match_score"] = round(float(row.get("average_rating") or 0.0), 4)
            results.append(record)
        return results

    def hybrid_recommendations(self, query: str, limit: int = 6, llm_provider: OpenAIProvider | None = None) -> list[dict]:
        normalized = self._normalize_query(query)
        if not normalized:
            return []

        provider = llm_provider or self.llm_provider
        preferences = {}
        if provider and getattr(provider, "available", False):
            preferences = provider.extract_preferences(normalized)

        search_terms = set(re.findall(r"[a-z0-9]+", normalized))
        preference_terms = set(search_terms)
        if isinstance(preferences, dict):
            preference_terms |= {token for field in ["genres", "topics", "authors"] for token in preferences.get(field, []) if isinstance(token, str) for token in re.findall(r"[a-z0-9]+", token.lower())}

        candidate_books = self.search_books(normalized, limit=max(limit * 4, 10))
        if not candidate_books:
            candidate_books = self.category_recommendations(normalized, limit=max(limit * 4, 10))
        if not candidate_books:
            return []

        scored = []
        for book in candidate_books:
            score = self._score_candidate_book(normalized, preference_terms, book)
            if isinstance(preferences, dict):
                score += 0.12 * len([field for field in ["genres", "topics", "authors"] if any(token.lower() in (book.get("categories") or "").lower() or token.lower() in (book.get("authors") or "").lower() for token in preferences.get(field, []))])
            scored.append({**book, "_score": score})

        ranked = sorted(scored, key=lambda item: item["_score"], reverse=True)
        unique_books = []
        seen = set()
        for result in ranked:
            title = str(result.get("title") or "").strip().lower()
            if not title or title in seen:
                continue
            seen.add(title)
            unique_books.append({key: value for key, value in result.items() if key != "_score"})
            if len(unique_books) >= limit:
                break

        return unique_books


if __name__ == "__main__":
    engine = BookRecommender()
    print(engine.recommend_for_book("The Hobbit", limit=3))
