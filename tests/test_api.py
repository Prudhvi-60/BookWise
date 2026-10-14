from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_homepage_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "BookWise" in response.text


def test_search_endpoint_returns_books():
    response = client.get("/api/search", params={"q": "hobbit"})
    assert response.status_code == 200
    data = response.json()
    assert "books" in data
    assert len(data["books"]) > 0


def test_chat_endpoint_accepts_messages():
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)


def test_static_assets_are_served():
    response = client.get("/static/js/app.js")

    assert response.status_code == 200
    assert "escapeHTML" in response.text


def test_favicon_request_does_not_404():
    assert client.get("/favicon.ico").status_code == 204


def test_invalid_search_and_chat_inputs_return_validation_errors():
    assert client.get("/api/search").status_code == 422
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 1001}).status_code == 422


def test_recommendations_require_a_query():
    response = client.get("/api/recommendations")

    assert response.status_code == 400
    assert response.json()["detail"]


def test_title_recommendations_exclude_the_selected_book():
    response = client.get("/api/recommendations", params={"title": "The Hobbit"})

    assert response.status_code == 200
    assert all(book["title"] != "The Hobbit" for book in response.json()["books"])
