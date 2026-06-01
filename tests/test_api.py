from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_inicio():
    response = client.get("/")

    assert response.status_code == 200
    assert "mensaje" in response.json()
    assert "modelos_disponibles" in response.json()

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert "estado" in response.json()
    assert "modelos" in response.json()
    assert "logistic_regression" in response.json()["modelos"]
    assert "random_forest" in response.json()["modelos"]

