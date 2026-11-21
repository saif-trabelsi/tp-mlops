"""Étape 2 : contrôle de qualité des données (séance 10). Arrête le pipeline si le lot est suspect."""
import sys
import pandas as pd

COLONNES_ATTENDUES = 14 # 13 features + la classe
SEUIL_NULLS = 0.05 # au plus 5 % de valeurs manquantes par colonne
MIN_LIGNES = 150

def valider(chemin: str) -> list[str]:
    """Renvoie la liste des problèmes trouvés (vide si tout va bien)."""
    df = pd.read_csv(chemin)
    problemes = []
    if df.shape[1] != COLONNES_ATTENDUES:
        problemes.append(f"{df.shape[1]} colonnes au lieu de {COLONNES_ATTENDUES}")
    if len(df) < MIN_LIGNES:
        problemes.append(f"lot trop petit : {len(df)} lignes")
    taux = df.isna().mean()
    for col, t in taux[taux > SEUIL_NULLS].items():
        problemes.append(f"{col} : {t:.0%} de valeurs manquantes")
    if (df.drop(columns="classe") < 0).any().any():
        problemes.append("valeurs négatives impossibles pour des mesures chimiques")
    if not set(df["classe"].dropna().unique()) <= {0, 1, 2}:
        problemes.append("classe inconnue")
    return problemes

if __name__ == "__main__":
    erreurs = valider(sys.argv[1] if len(sys.argv) > 1 else "data/vins.csv")
    for e in erreurs:
        print("ERREUR QUALITÉ :", e)
    sys.exit(1 if erreurs else 0) # code 1 = échec : DVC et la CI s'arrêtent ici
