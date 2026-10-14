from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_processing import clean_dataset, get_dataset_path, load_dataset
from src.recommender import BookRecommender


def evaluate_catalogue(dataset_path: str | Path | None = None) -> dict:
    df = clean_dataset(load_dataset(dataset_path or get_dataset_path()))
    categories = df["categories"].fillna("").astype(str).str.split(",").explode().str.strip()
    categories = categories[categories.ne("")]
    searchable = df["summary_text"].fillna("").str.strip().ne("")
    return {
        "row_count": int(len(df)),
        "missing_titles": int(df["title"].isna().sum()),
        "missing_ratings": int(df["average_rating"].isna().sum()),
        "missing_thumbnails": int(df["thumbnail"].isna().sum()),
        "distinct_categories": int(categories.nunique()),
        "searchable_books": int(searchable.sum()),
    }


def evaluate_recommendations(recommender: BookRecommender) -> dict:
    results = []
    recommended_ids: set[str] = set()
    for title in ["The Hobbit", "Pride and Prejudice", "1984", "The Hunger Games"]:
        recs = recommender.recommend_for_book(title, limit=5)
        recommended_ids.update(
            str(book.get("isbn13") or book.get("title"))
            for book in recs
            if book.get("isbn13") or book.get("title")
        )
        results.append({
            "query": title,
            "count": len(recs),
            "unique_titles": len({book.get("title", "") for book in recs}),
            "excluded_query": all(book.get("title") != title for book in recs),
        })

    diversity = sum(item["unique_titles"] for item in results) / max(len(results), 1)
    return {
        "sample_evaluation": results,
        "average_unique_titles": round(diversity, 2),
        "sampled_catalogue_coverage": round(len(recommended_ids) / max(len(recommender.df), 1), 4),
        "notes": "Coverage is the share of catalogue entries returned across four fixed seed books. No relevance labels are available, so this is a mechanical retrieval check, not a relevance or supervised-classification score.",
    }


def main() -> None:
    report = evaluate_catalogue()
    engine = BookRecommender()
    print(report)
    print(evaluate_recommendations(engine))


if __name__ == "__main__":
    main()
