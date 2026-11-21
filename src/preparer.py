"""Étape 1 du pipeline : produire le jeu de données de référence data/vins.csv."""
from pathlib import Path
from sklearn.datasets import load_wine

Path("data").mkdir(exist_ok=True)
vins = load_wine(as_frame=True).frame # 178 vins, 13 mesures chimiques + la classe (0, 1, 2)
vins = vins.rename(columns={"target": "classe"})
vins.to_csv("data/vins.csv", index=False)
print(f"{len(vins)} lignes, {vins.shape[1]} colonnes écrites dans data/vins.csv")
