"""Enregistre le meilleur run dans le registre et lui donne l'alias « champion »."""
from pathlib import Path
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
run_id = Path("modeles/run_id.txt").read_text().strip()
version = mlflow.register_model(f"runs:/{run_id}/modele", "classifieur-vins") # crée la version N
client = MlflowClient()
client.set_registered_model_alias("classifieur-vins", "champion", version.version)
client.update_model_version("classifieur-vins", version.version,
                            description="RandomForest choisi par validation croisée (TP MLOps, partie 2)")
print(f"classifieur-vins v{version.version} -> alias @champion")

# Vérification : on recharge le modèle PAR SON ALIAS, comme le fera l'API
modele = mlflow.pyfunc.load_model("models:/classifieur-vins@champion")
print("Rechargé depuis le registre :", type(modele).__name__)
