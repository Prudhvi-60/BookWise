# BookWise — Discover Your Next Read

BookWise is a FastAPI book-discovery application. Search, category browsing, and recommendations are grounded in the local catalogue. A supplied intents file drives deterministic chatbot greetings and search behavior. Optional OpenAI preference extraction is disabled unless credentials are configured.

## Architecture

- `main.py`: FastAPI application, API routes, templates, and static assets.
- `src/data_processing.py`: repository-relative CSV loading, ISBN-preserving parsing, validation, cleaning, statistics, and EDA output.
- `src/recommender.py`: sparse TF-IDF search and cosine similarity; category overlap and ratings are secondary ranking signals.
- `src/chatbot.py`: deterministic matching against `data/raw/intents.json` and catalogue-backed responses.
- `src/llm_provider.py`: optional OpenAI preference extraction with validated output and deterministic fallback.
- `src/evaluation.py`: reproducible mechanical retrieval checks; the catalogue contains no ground-truth relevance labels.
- `templates/`, `static/`: Jinja2 page, CSS, and browser interactions.

The app currently does not use OpenAI tool/function calling. It does not generate or invent catalogue entries. No supervised purchase classifier or classification metrics are claimed.

## Data

The application reads `data/raw/data.csv` and `data/raw/intents.json`, relative to the project directory. The catalogue schema is `isbn13`, `isbn10`, `title`, `subtitle`, `authors`, `categories`, `thumbnail`, `description`, `published_year`, `average_rating`, `num_pages`, and `ratings_count`. Cleaned data is reproducibly written to `data/processed/cleaned_books.csv`; the raw CSV is not modified.

The current checkout contains a small catalogue; see `reports/eda/eda_summary.md` for measured dimensions, missingness, duplicates, and distributions. The original data provider’s licensing and redistribution terms still need to be verified before publishing the dataset.

## Local Setup (Windows PowerShell)

From the repository root:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

LLM preference extraction is optional. To enable it for a local session, set credentials in the current PowerShell process (do not commit them):

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_MODEL = "gpt-4o-mini"
```

Run tests and start the server:

```powershell
python -c "import sys; print(sys.executable); import main, src; print('Imports OK')"
python -m pytest -q
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000`. API documentation is available at `/docs`.

## API

- `GET /`: BookWise discovery page.
- `GET /health`: readiness, catalogue size, and optional LLM availability.
- `GET /api/stats`: computed catalogue dimensions and summary statistics.
- `GET /api/search?q=...`: title, author, category, description, and TF-IDF search.
- `GET /api/categories`: available catalogue categories.
- `GET /api/recommendations?title=...&limit=6`: similar books; also accepts `q` or `genre`.
- `POST /api/chat`: JSON body `{"message":"..."}`.
- `GET /docs`: interactive OpenAPI documentation.

Search text is limited to 200 characters; chat messages are limited to 1,000 characters and result limits are bounded.

## Evaluation and Tests

Run:

```powershell
python -m src.evaluation
python -m pytest -q
```

Evaluation checks result validity, self-exclusion, unique titles, and sampled catalogue coverage for representative seed books. There are no explicit relevance labels, so no precision, recall, F1, or accuracy claims are appropriate.

## Render Deployment

The root `render.yaml` configures a Python web service with:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health-check path: `/health`

To deploy, push the project and required data files to a GitHub repository, create a Render Blueprint or web service from that repository, and verify the build/deploy logs and `/health` response. Set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in Render’s environment only if optional OpenAI preference extraction is wanted. No remote deployment has been performed from this workspace.

## Troubleshooting

- Run commands from the repository root and use the project venv’s Python to avoid missing dependency/import errors.
- If startup reports a missing catalogue or intents file, verify both files are included at the repository-relative paths above.
- A `TestClient` AnyIO deprecation warning may appear with the currently pinned FastAPI/Starlette test stack; it does not affect the app’s HTTP routes.
- Cover URLs are catalogue data and may fail at their origin; the UI displays a local unavailable-cover placeholder.
