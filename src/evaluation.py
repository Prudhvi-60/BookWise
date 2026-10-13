from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_processing import clean_dataset, get_dataset_path, load_dataset
from src.recommender import BookRecommender


def evaluate_catalogue(dataset_path: str | Path | None = None) -> dict:
    df = clean_dataset(load_dataset(dataset_path or get_dataset_path()))
    return {
        "row_count": int(len(df)),
        "missing_titles": int(df["title"].isna().sum()),
        "missing_ratings": int(df["average_rating"].isna().sum()),
        "missing_thumbnails": int(df["thumbnail"].isna().sum()),
        "distinct_categories": int(df["categories"].fillna("").str.split(",").explode().str.strip().ne("").nunique()),
        "coverage": float(1.0 if len(df) else 0.0),
    }


def evaluate_recommendations(recommender: BookRecommender) -> dict:
    results = []
    for title in ["The Hobbit", "Pride and Prejudice", "1984", "The Hunger Games"]:
        recs = recommender.recommend_for_book(title, limit=5)
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
        "notes": "No explicit rating labels were available; relevance is based on textual similarity, category overlap, and rating quality rather than supervised accuracy metrics.",
    }


def main() -> None:
    report = evaluate_catalogue()
    engine = BookRecommender()
    print(report)
    print(evaluate_recommendations(engine))


if __name__ == "__main__":
    main()
