"""Tests du contrat de l'API (exécutés en CI, partie 5)."""
import os
os.environ.setdefault("MODEL_PATH", "modeles/modele_final.pkl") # AVANT l'import de l'application
from fastapi.testclient import TestClient # noqa: E402
from api.main import app # noqa: E402

client = TestClient(app)
VIN = [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050] # exemple du cours

def test_prediction_valide():
    r = client.post("/predict", json={"features": VIN})
    assert r.status_code == 200
    corps = r.json()
    assert corps["prediction"] in (0, 1, 2)
    assert 0.0 <= corps["confiance"] <= 1.0

def test_vin_connu_bien_classe():
    # ce vin est le premier du jeu Wine : il appartient à la classe 0
    assert client.post("/predict", json={"features": VIN}).json()["prediction"] == 0

def test_entree_incomplete_refusee():
    assert client.post("/predict", json={"features": [1.0, 2.0]}).status_code == 422

def test_health():
    assert client.get("/health").json() == {"status": "ok"}
