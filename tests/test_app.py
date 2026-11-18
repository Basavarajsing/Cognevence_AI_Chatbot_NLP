"""
Integration Tests for Flask Web Application Endpoints.
Verifies routes, session management, input validation, export features,
analytics computation, and error handlers.
"""

import json
import pytest
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Cognevance NLP" in response.data
    assert b"chatMessages" in response.data


def test_chat_endpoint_valid_message(client):
    payload = {"message": "Hello, how are you?"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "response" in data
    assert "intent" in data
    assert "confidence" in data
    assert "timestamp" in data
    assert data["confidence"] >= 0.0


def test_chat_endpoint_empty_message(client):
    payload = {"message": "   "}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["is_fallback"] is True
    assert data["intent"] == "fallback"


def test_chat_endpoint_missing_message_field(client):
    response = client.post("/chat", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_chat_endpoint_oversized_message(client):
    oversized = "a" * 1005
    response = client.post("/chat", json={"message": oversized})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_suggested_endpoint(client):
    response = client.get("/suggested")
    assert response.status_code == 200
    data = response.get_json()
    assert "suggestions" in data
    assert len(data["suggestions"]) > 0


def test_clear_endpoint(client):
    # First send a message to populate history
    client.post("/chat", json={"message": "Hi"})
    
    # Clear history
    response = client.post("/clear")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"


def test_dashboard_route(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Chat Analytics" in response.data


def test_api_analytics_endpoint(client):
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.get_json()
    assert "total_messages" in data
    assert "total_queries" in data
    assert "avg_confidence" in data
    assert "fallback_rate" in data
    assert "model_metrics" in data


def test_export_json_endpoint(client):
    # Chat once to ensure content
    client.post("/chat", json={"message": "What is Python?"})
    
    response = client.get("/export/json")
    assert response.status_code == 200
    assert response.mimetype == "application/json"
    assert "Content-Disposition" in response.headers
    
    data = json.loads(response.data)
    assert "conversation" in data
    assert len(data["conversation"]) >= 2


def test_export_txt_endpoint(client):
    # Chat once to ensure content
    client.post("/chat", json={"message": "What is NLP?"})
    
    response = client.get("/export/txt")
    assert response.status_code == 200
    assert response.mimetype == "text/plain"
    assert b"AI CHATBOT CONVERSATION TRANSCRIPT" in response.data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
