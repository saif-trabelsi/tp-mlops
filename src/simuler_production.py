"""Fabrique un lot de « production » à partir du jeu de test, avec une dérive volontaire."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
vins = pd.read_csv("data/vins.csv").drop(columns="classe")

normal = vins.sample(120, random_state=1) # production sans dérive
derive = vins.sample(120, random_state=2).copy()
derive["alcohol"] += 1.2 # vins plus alcoolisés (millésime chaud)
derive["color_intensity"] *= 1.5 # nouvelle méthode de vinification
derive["proline"] += rng.normal(0, 30, len(derive)) # bruit de mesure d'un nouveau capteur

normal.to_csv("data/production_normale.csv", index=False)
derive.to_csv("data/production_derive.csv", index=False)
print("Lots écrits : production_normale.csv, production_derive.csv")
