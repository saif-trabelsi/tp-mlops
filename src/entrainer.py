"""Étape 3 : entraîner 9 variantes, tout tracer dans MLflow, sauvegarder la meilleure (TP 4 du cours)."""
import hashlib
import itertools
import json
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

P = yaml.safe_load(open("params.yaml"))["entrainement"]
mlflow.set_tracking_uri("sqlite:///mlflow.db") # base SQL : indispensable pour le registre
mlflow.set_experiment("vins-classification")

vins = pd.read_csv("data/vins.csv")
empreinte = hashlib.md5(open("data/vins.csv", "rb").read()).hexdigest() # version des données
X, y = vins.drop(columns="classe"), vins["classe"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=P["test_size"],
                                          random_state=P["seed"], stratify=y)

resultats = []
for n, d in itertools.product(P["n_estimators"], P["max_depth"]): # 3 x 3 = 9 runs
    with mlflow.start_run(run_name=f"rf-{n}-{d}") as run:
        mlflow.set_tag("donnees_md5", empreinte)
        mlflow.log_params({"n_estimators": n, "max_depth": d, "seed": P["seed"]})
        modele = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=P["seed"])
        cv = cross_val_score(modele, X_tr, y_tr, cv=5).mean() # robuste sur un petit jeu
        modele.fit(X_tr, y_tr)
        metriques = {"acc_train": float(modele.score(X_tr, y_tr)),
                     "acc_test": float(modele.score(X_te, y_te)),
                     "cv_mean": float(cv)}
        mlflow.log_metrics(metriques)
        mlflow.sklearn.log_model(modele, "modele", input_example=X_te.head(2))
        resultats.append((cv, -n, run.info.run_id, modele, metriques, n, d))

# Meilleur : plus grande moyenne de validation croisée ; à égalité, la forêt la plus petite
cv, _, run_id, modele, metriques, n, d = max(resultats, key=lambda r: (r[0], r[1]))
Path("modeles").mkdir(exist_ok=True)
joblib.dump(modele, "modeles/modele_final.pkl")
Path("modeles/run_id.txt").write_text(run_id)
json.dump(metriques, open("metriques.json", "w"), indent=2)
print(f"Meilleur run : n_estimators={n}, max_depth={d}, accuracy test={metriques['acc_test']:.3f}")
