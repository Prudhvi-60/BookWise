from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_processing import clean_dataset, get_dataset_path, load_dataset


class BookRecommender:
    def __init__(self, dataset_path: str | None = None):
        df = clean_dataset(load_dataset(dataset_path or get_dataset_path()))
        self.df = df.copy().reset_index(drop=True)
        self.df["search_text"] = (
            self.df[["title", "subtitle", "authors", "categories", "description"]].fillna("").astype(str).agg(lambda row: " ".join(row), axis=1)
        )
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.df["search_text"].fillna(""))
        self._title_lookup = {str(title).strip().lower(): idx for idx, title in enumerate(self.df["title"].fillna(""))}
        self._isbn_lookup = {str(isbn).strip(): idx for isbn, idx in zip(self.df["isbn13"].fillna(""), range(len(self.df)))}

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
    def _as_book_record(row: pd.Series) -> dict:
        return {
            "isbn13": row.get("isbn13"),
            "isbn10": row.get("isbn10"),
            "title": str(row.get("title") or "").strip() or None,
            "subtitle": row.get("subtitle"),
            "authors": row.get("authors"),
            "categories": row.get("categories"),
            "thumbnail": row.get("thumbnail"),
            "description": row.get("description"),
            "published_year": row.get("published_year"),
            "average_rating": None if pd.isna(row.get("average_rating")) else row.get("average_rating"),
            "num_pages": None if pd.isna(row.get("num_pages")) else row.get("num_pages"),
            "ratings_count": None if pd.isna(row.get("ratings_count")) else row.get("ratings_count"),
        }

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
            results.append(self._as_book_record(row))
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
            results.append(self._as_book_record(row))
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
        return [self._as_book_record(row) for _, row in subset.head(limit).iterrows()]


if __name__ == "__main__":
    engine = BookRecommender()
    print(engine.recommend_for_book("The Hobbit", limit=3))
