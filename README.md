# BookWise — Intelligent Book Recommendation System

BookWise is a FastAPI-based book discovery app that combines a dataset-backed recommendation engine with optional LLM-assisted preference understanding. The project keeps the original catalogue-driven workflow, extends it with natural-language recommendation input, and falls back safely when the LLM is unavailable.

## Architecture discovered

The repository is structured around the following components:

- `main.py`: FastAPI application entry point.
- `src/data_processing.py`: dataset loading, schema validation, cleaning, and summary stats.
- `src/recommender.py`: TF-IDF search, similarity-based recommendations, category filtering, and hybrid ranking.
- `src/chatbot.py`: intent-based conversational helper using the local dataset and keyword patterns.
- `src/llm_provider.py`: OpenAI-compatible provider abstraction for preference extraction and reasoning.
- `templates/index.html` and `static/js/app.js`: browser UI for search and recommendations.
- `data/raw/data.csv`: catalogue data used for metadata-based recommendations.
- `data/raw/intents.json`: static chat intents for the bot.

## Existing stack

- Python 3.12+
- FastAPI
- Jinja2 + static frontend
- Pandas + NumPy
- scikit-learn for TF-IDF and cosine similarity
- pytest for automated tests
- OpenAI Python SDK for optional LLM integration
- python-dotenv for environment loading

## Recommendation flow

The original system used a content-based approach:

- normalize and clean the source dataset,
- create TF-IDF features from title, subtitle, authors, categories, and description,
- search by literal title/author/category matches or semantic similarity,
- rank books by match score, average rating, and popularity.

The upgrade adds a hybrid path:

1. Validate the user query.
2. Optional LLM extraction of genres, interests, reading level, tone, goals, exclusions, and requested count.
3. Candidate retrieval from the catalog using the existing search and similarity logic.
4. Ranking with dataset metadata, category/keyword overlap, and popularity/ratings.
5. Final validation against real catalog entries.
6. Safe fallback to dataset-only recommendations when the LLM is unavailable or errors occur.

## Environment setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the sample environment file and set your key:

```bash
copy .env.example .env
```

4. Add your API key and model in `.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

5. Start the application:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

6. Open http://localhost:8000

## API configuration and behavior

- The app reads environment variables from `.env` using `python-dotenv`.
- LLM functionality is optional; missing credentials do not crash the app.
- When no key is configured, the project continues using the dataset recommendation engine.
- Model outputs are validated before use; malformed LLM responses trigger safe fallback behavior.

## Testing

Run the automated tests with:

```bash
pytest -q
```

The suite covers:

- dataset loading and cleaning,
- recommendation and search behavior,
- API endpoints,
- chatbot fallback responses,
- LLM provider behavior,
- hybrid fallback behavior.

## Deployment

This project is prepared for Render with the existing `render.yaml` configuration:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Ensure the environment exposes `OPENAI_API_KEY` and `OPENAI_MODEL` in the deployment environment, and avoid committing secrets into source control.

## Known limitations

- The catalogue is small and domain-specific; recommendations are grounded to the supplied catalog only.
- The LLM is used for preference extraction and explanation, not to invent books.
- The app does not support a vector database or advanced embeddings because the dataset is small and the existing architecture is lightweight.
- Real LLM requests require a valid API key and network access; tests use mocked provider responses and do not require paid API credits.

## Licensing note

The source data and intent definitions should be reviewed for their original licensing or redistribution terms before public deployment or redistribution.
