from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data" / "raw"
PROCESSED_DIR = APP_ROOT / "data" / "processed"


def _normalize_isbn(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", ""}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def get_dataset_path(filename: str = "data.csv") -> Path:
    candidates = [
        APP_ROOT / "data" / "raw" / filename,
        APP_ROOT / "data" / filename,
        APP_ROOT / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return DATA_DIR / filename


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path else get_dataset_path()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    df = pd.read_csv(
        dataset_path,
        keep_default_na=True,
        dtype={"isbn13": "string", "isbn10": "string"},
    )
    return df.copy()


def normalize_text_series(series: pd.Series) -> pd.Series:
    text = series.map(lambda value: "" if value is None or pd.isna(value) else str(value))
    text = text.str.strip().str.replace(r"\s+", " ", regex=True)
    return text.replace({"nan": "", "None": "", "NaN": ""})


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    required_columns = {
        "isbn13",
        "isbn10",
        "title",
        "subtitle",
        "authors",
        "categories",
        "thumbnail",
        "description",
        "published_year",
        "average_rating",
        "num_pages",
        "ratings_count",
    }
    missing_columns = required_columns - set(cleaned.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    for column in ["title", "subtitle", "authors", "categories", "thumbnail", "description"]:
        cleaned[column] = normalize_text_series(cleaned[column]).replace({"nan": "", "None": "", "NaN": ""})

    cleaned["isbn13"] = cleaned["isbn13"].map(_normalize_isbn)
    cleaned["isbn10"] = cleaned["isbn10"].map(_normalize_isbn)

    cleaned["published_year"] = pd.to_numeric(cleaned["published_year"], errors="coerce")
    cleaned["published_year"] = cleaned["published_year"].where(cleaned["published_year"].between(0, 2035), pd.NA)

    cleaned["average_rating"] = pd.to_numeric(cleaned["average_rating"], errors="coerce")
    cleaned["average_rating"] = cleaned["average_rating"].where(cleaned["average_rating"].between(0, 5), pd.NA)

    cleaned["num_pages"] = pd.to_numeric(cleaned["num_pages"], errors="coerce")
    cleaned["num_pages"] = cleaned["num_pages"].where(cleaned["num_pages"].gt(0), pd.NA)

    cleaned["ratings_count"] = pd.to_numeric(cleaned["ratings_count"], errors="coerce")
    cleaned["ratings_count"] = cleaned["ratings_count"].where(cleaned["ratings_count"].ge(0), pd.NA)

    cleaned["title"] = cleaned["title"].map(lambda value: "" if value is None or pd.isna(value) else str(value).strip())
    cleaned["authors"] = cleaned["authors"].map(lambda value: "" if value is None or pd.isna(value) else str(value).strip())
    cleaned["categories"] = cleaned["categories"].map(lambda value: "" if value is None or pd.isna(value) else str(value).strip())

    cleaned = cleaned.drop_duplicates().copy()

    cleaned["title_clean"] = cleaned["title"].fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    cleaned["author_clean"] = cleaned["authors"].fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()

    cleaned = cleaned.sort_values(by=["title", "authors", "published_year"], na_position="last").drop_duplicates(subset=["isbn13"], keep="last")
    cleaned = cleaned.sort_values(by=["title", "authors"], na_position="last").drop_duplicates(subset=["title_clean"], keep="last")

    for column in ["title", "subtitle", "authors", "categories", "description", "thumbnail"]:
        cleaned[column] = cleaned[column].replace({"": pd.NA})

    cleaned["categories"] = cleaned["categories"].fillna("")
    cleaned["description"] = cleaned["description"].fillna("")
    cleaned["summary_text"] = cleaned[["title", "subtitle", "authors", "categories", "description"]].fillna("").astype(str).agg(" ".join, axis=1)
    cleaned["summary_text"] = cleaned["summary_text"].str.replace(r"\s+", " ", regex=True).str.strip()

    return cleaned.reset_index(drop=True)


def get_dataset_stats(df: pd.DataFrame) -> dict:
    categories = (
        df["categories"].fillna("").astype(str).str.split(",").explode().str.strip()
        if "categories" in df
        else pd.Series(dtype="string")
    )
    ratings = pd.to_numeric(df["average_rating"], errors="coerce") if "average_rating" in df else pd.Series(dtype="float64")
    stats = {
        "rows": int(len(df)),
        "category_count": int(categories[categories.ne("")].nunique()),
        "average_rating": None if ratings.dropna().empty else round(float(ratings.mean()), 2),
        "columns": list(df.columns),
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "missing_pct": {col: round(float(df[col].isna().mean() * 100), 2) for col in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return stats


def save_processed_dataset(df: pd.DataFrame, filename: str = "cleaned_books.csv") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path


def prepare_dataset() -> pd.DataFrame:
    raw_df = load_dataset(get_dataset_path())
    cleaned = clean_dataset(raw_df)
    save_processed_dataset(cleaned)
    return cleaned


def generate_eda_report(df: pd.DataFrame | None = None, output_dir: str | Path | None = None) -> Path:
    """Generate a minimal EDA artifact set for the project reports folder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    source = df.copy() if df is not None else load_dataset()
    books = clean_dataset(source)
    report_dir = Path(output_dir) if output_dir is not None else APP_ROOT / "reports" / "eda"
    report_dir.mkdir(parents=True, exist_ok=True)

    missing_values = source.isna().sum()
    missing_summary = [
        f"- {column}: {int(count)} ({count / max(len(source), 1) * 100:.1f}%)"
        for column, count in missing_values.items()
        if count
    ] or ["- None"]
    normalized_titles = source["title"].fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    duplicate_isbn13 = int(source["isbn13"].dropna().duplicated().sum())
    duplicate_isbn10 = int(source["isbn10"].dropna().duplicated().sum())
    duplicate_titles = int(normalized_titles[normalized_titles.ne("")].duplicated().sum())
    numeric_summaries = []
    for column in ["published_year", "average_rating", "num_pages", "ratings_count"]:
        values = pd.to_numeric(source[column], errors="coerce")
        numeric_summaries.append(
            f"- {column}: valid={int(values.notna().sum())}, min={values.min()}, median={values.median()}, max={values.max()}"
        )

    summary = [
        "# BookWise Exploratory Data Analysis",
        f"- Source dimensions: {len(source)} rows x {len(source.columns)} columns",
        f"- Cleaned rows: {len(books)} (removed {len(source) - len(books)} records)",
        f"- Exact duplicate source rows: {int(source.duplicated().sum())}",
        f"- Duplicate ISBN-13 values: {duplicate_isbn13}",
        f"- Duplicate ISBN-10 values: {duplicate_isbn10}",
        f"- Duplicate normalized titles: {duplicate_titles}",
        "",
        "## Missing values in source data",
        *missing_summary,
        "",
        "## Numeric distributions",
        *numeric_summaries,
        "",
        "## Text distributions",
        f"- Distinct non-empty categories: {int(source['categories'].fillna('').astype(str).str.split(',').explode().str.strip().replace('', pd.NA).nunique())}",
        f"- Distinct non-empty authors: {int(source['authors'].fillna('').astype(str).str.strip().replace('', pd.NA).nunique())}",
        "",
        "## Cleaning decisions",
        "- Read ISBN-10 and ISBN-13 as strings, preserving leading zeroes.",
        "- Normalized whitespace and converted empty textual fields to missing values.",
        "- Kept records with missing optional text fields; invalid numeric ranges become missing, not fabricated values.",
        "- Removed exact duplicates and then duplicate ISBN-13/title keys deterministically, retaining the last sorted record.",
    ]
    (report_dir / "eda_summary.md").write_text("\n".join(summary), encoding="utf-8")

    if not books.empty:
        top_categories = books["categories"].fillna("").astype(str).str.split(",").explode().str.strip()
        top_categories = top_categories[top_categories != ""].value_counts().head(8)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_categories.values, y=top_categories.index, color="#5f49d3", ax=ax)
        ax.set_title("Top categories")
        ax.set_xlabel("Count")
        ax.set_ylabel("Category")
        fig.tight_layout()
        fig.savefig(report_dir / "top_categories.png", dpi=150)
        plt.close(fig)

        ratings = books["average_rating"].dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(ratings, bins=15, color="purple", ax=ax)
        ax.set_title("Average rating distribution")
        ax.set_xlabel("Average rating")
        ax.set_ylabel("Books")
        fig.tight_layout()
        fig.savefig(report_dir / "rating_distribution.png", dpi=150)
        plt.close(fig)

    return report_dir
