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
