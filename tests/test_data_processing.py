import pandas as pd

from src.data_processing import clean_dataset, get_dataset_path, load_dataset


def test_load_dataset_reads_expected_columns():
    df = load_dataset(get_dataset_path())
    assert isinstance(df, pd.DataFrame)
    required = {
        "isbn13", "isbn10", "title", "subtitle", "authors", "categories",
        "thumbnail", "description", "published_year", "average_rating",
        "num_pages", "ratings_count"
    }
    assert required.issubset(set(df.columns))
    assert len(df) > 0


def test_clean_dataset_handles_missing_values_and_normalizes_text():
    df = load_dataset(get_dataset_path())
    cleaned = clean_dataset(df)
    assert len(cleaned) <= len(df)
    assert cleaned["title"].notna().all()
    assert cleaned["authors"].notna().all()
    assert cleaned["categories"].notna().all()
    assert cleaned["isbn13"].astype(str).str.len().gt(0).all()


def test_load_dataset_preserves_isbn_leading_zeroes(tmp_path):
    csv_path = tmp_path / "books.csv"
    csv_path.write_text(
        "isbn13,isbn10,title,subtitle,authors,categories,thumbnail,description,published_year,average_rating,num_pages,ratings_count\n"
        "9780000000001,012345678X,Example,,,,,2000,,,\n",
        encoding="utf-8",
    )

    loaded = load_dataset(csv_path)

    assert loaded.loc[0, "isbn13"] == "9780000000001"
    assert loaded.loc[0, "isbn10"] == "012345678X"
