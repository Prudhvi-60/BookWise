# BookWise — Intelligent Book Recommendation System

BookWise is a lightweight book recommendation and discovery platform built with FastAPI and a content-based recommender. It combines a cleaned catalogue, a TF-IDF similarity engine, an intents-based chatbot, and a responsive frontend.

## Project goals

- Recommend books from a catalogue using content similarity.
- Support title, author, category, and keyword searches.
- Provide a lightweight chatbot powered by the supplied intent definitions.
- Expose a modern web UI and clean REST API.
- Run locally and prepare for deployment on Render.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
4. Open http://localhost:8000

## Data

The application expects a catalogue CSV with the following columns:

- `isbn13`
- `isbn10`
- `title`
- `subtitle`
- `authors`
- `categories`
- `thumbnail`
- `description`
- `published_year`
- `average_rating`
- `num_pages`
- `ratings_count`

It also expects an `intents.json` file in `data/raw/` with the provided chatbot intent definitions.

## Recommendation method

The recommender uses a content-based strategy built on TF-IDF vectorization applied to the title, subtitle, authors, categories, and description fields. Similarity is computed with cosine similarity and then adjusted with category overlap and rating quality, without letting missing or sparse ratings act like a zero rating.

## Evaluation

No explicit supervised labels were available in the supplied dataset, so the recommender is evaluated qualitatively and mechanically using:

- valid and unique results,
- exclusion of the query book,
- coverage of catalogue items,
- category and author matching,
- diversity checks,
- empty and unknown query handling.

## Testing

Run the project tests with:

```bash
pytest -q
```

## Deployment notes

This project is set up for Render with the standard FastAPI runtime pattern:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

The app uses repository-relative paths and avoids any machine-specific file references.

## Licensing note

The source data and intent files should be checked for their original licensing or redistribution terms before public deployment or redistribution.
