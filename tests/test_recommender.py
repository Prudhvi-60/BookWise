from src.recommender import BookRecommender


def test_recommendations_exclude_query_book_and_return_valid_records():
    engine = BookRecommender()
    books = engine.recommend_for_book("The Hobbit", limit=5)
    assert len(books) > 0
    assert all(book["title"] != "The Hobbit" for book in books)
    assert all("title" in book for book in books)
    assert all("authors" in book for book in books)


def test_search_returns_relevant_books_for_queries():
    engine = BookRecommender()
    books = engine.search_books("fantasy")
    assert len(books) > 0
    assert any("Fantasy" in (book.get("categories", "") or "") or "fantasy" in (book.get("title", "") or "").lower() for book in books)


def test_unknown_queries_are_handled_gracefully():
    engine = BookRecommender()
    books = engine.search_books("zzzz-not-a-real-book-title")
    assert books == []
