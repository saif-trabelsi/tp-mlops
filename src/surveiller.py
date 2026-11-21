"""Compare un lot de production à la référence : PSI et test de Kolmogorov-Smirnov par feature."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

SEUIL_PSI = 0.25 # > 0,25 : dérive importante (séance 13)
SEUIL_P = 0.01 # p-value du test KS

def psi(reference, production, classes=10):
    """Population Stability Index ; classes bâties sur les quantiles de la référence."""
    bornes = np.unique(np.quantile(reference, np.linspace(0, 1, classes + 1)))
    bornes[0], bornes[-1] = -np.inf, np.inf
    p_ref = np.histogram(reference, bornes)[0] / len(reference)
    p_prod = np.histogram(production, bornes)[0] / len(production)
    p_ref, p_prod = np.clip(p_ref, 1e-4, None), np.clip(p_prod, 1e-4, None) # évite ln(0)
    return float(np.sum((p_prod - p_ref) * np.log(p_prod / p_ref)))

def rapport(chemin_prod: str) -> pd.DataFrame:
    ref = pd.read_csv("data/vins.csv").drop(columns="classe")
    prod = pd.read_csv(chemin_prod)
    lignes = []
    for col in ref.columns:
        stat, p = ks_2samp(ref[col], prod[col])
        lignes.append({"feature": col, "psi": round(psi(ref[col], prod[col]), 3),
                       "ks": round(stat, 3), "p_value": p})
    df = pd.DataFrame(lignes).sort_values("psi", ascending=False)
    df["derive"] = (df["psi"] > SEUIL_PSI) | (df["p_value"] < SEUIL_P)
    return df

if __name__ == "__main__":
    r = rapport(sys.argv[1])
    print(r.head(6).to_string(index=False, formatters={"p_value": "{:.1e}".format}))
    derivees = r.loc[r["derive"], "feature"].tolist()
    if derivees:
        print(f"ALERTE : dérive détectée sur {derivees} -> déclencher le ré-entraînement")
        sys.exit(2) # code dédié : le pipeline CT sait qu'il faut ré-entraîner
    print("Aucune dérive significative.")
