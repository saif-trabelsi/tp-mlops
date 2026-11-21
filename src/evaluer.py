"""Porte de qualité (séance 14) : le nouveau modèle doit dépasser un seuil absolu ET égaler le champion."""
import json
import sys

SEUIL_ABSOLU = 0.90
nouveau = json.load(open("metriques.json")) # produit par src/entrainer.py
champion = json.load(open("metriques_champion.json")) # métriques du modèle en production

print(f"cv_mean nouveau = {nouveau['cv_mean']:.3f} | champion = {champion['cv_mean']:.3f}")

if nouveau["cv_mean"] < SEUIL_ABSOLU:
    sys.exit(f"REFUS : {nouveau['cv_mean']:.3f} < seuil absolu {SEUIL_ABSOLU}")
if nouveau["cv_mean"] < champion["cv_mean"] - 0.005: # tolérance au bruit de mesure
    sys.exit("REFUS : moins bon que le champion, il reste en production")
print("ACCEPTÉ : le nouveau modèle peut être déployé")
