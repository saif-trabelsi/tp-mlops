"""API de classification des vins (TP 5 du cours, complétée)."""
import logging
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Ordre des 13 mesures, identique à celui de l'entraînement (noms des colonnes de data/vins.csv)
FEATURES = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols",
            "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue",
            "od280/od315_of_diluted_wines", "proline"]

VERSION_MODELE = os.getenv("MODEL_VERSION", "1")
app = FastAPI(title="API de classification - TP MLOps", version="1.0.0")
modele = joblib.load(os.getenv("MODEL_PATH", "modele_final.pkl")) # chargé UNE fois, au démarrage
journal = logging.getLogger("uvicorn.error")

class Entree(BaseModel):
    """Corps de la requête : exactement 13 nombres (sinon FastAPI répond 422)."""
    features: list[float] = Field(min_length=13, max_length=13)

@app.post("/predict")
def predict(entree: Entree):
    donnees = pd.DataFrame([entree.features], columns=FEATURES) # mêmes noms qu'à l'entraînement
    probabilites = modele.predict_proba(donnees)[0]
    reponse = {"prediction": int(probabilites.argmax()), # classe 0, 1 ou 2
               "confiance": round(float(probabilites.max()), 3), # probabilité de cette classe
               "version_modele": VERSION_MODELE}
    journal.info("prediction %s entree=%s", reponse, entree.features) # matière pour le monitoring
    return reponse

@app.get("/health")
def health():
    """Sonde de santé : Docker, Kubernetes et le script de supervision l'interrogent."""
    return {"status": "ok"}
