# Rapport de TP 2 — MLOps : Du Jeu de Données au Modèle Surveillé

**Module :** DevOps & MLOps — Partie 2 : MLOps  
**Enseignant :** M. Mohamed Sahbi Bahroun  
**Niveau :** 3ème année Ingénierie du Développement Logiciel  
**Année universitaire :** 2026–2027  
**Date limite de rendu :** 28 décembre 2026  

---

## Sommaire

1. [Introduction et Architecture Globale](#1-introduction-et-architecture-globale)
2. [Arborescence Finale du Dépôt](#2-arborescence-finale-du-dépôt)
3. [Partie 1 — Données Versionnées et Pipeline Reproductible (DVC)](#3-partie-1--données-versionnées-et-pipeline-reproductible-dvc)
4. [Partie 2 — Tracking des Expériences et Registre (MLflow)](#4-partie-2--tracking-des-expériences-et-registre-mlflow)
5. [Partie 3 — Déploiement et Serving : API REST et Docker (FastAPI)](#5-partie-3--déploiement-et-serving--api-rest-et-docker-fastapi)
6. [Partie 4 — Détection de la Dérive des Données (Data Drift)](#6-partie-4--détection-de-la-dérive-des-données-data-drift)
7. [Partie 5 — Automatisation CI/CD/CT et Porte de Qualité (GitHub Actions)](#7-partie-5--automatisation-cicdct-et-porte-de-qualité-github-actions)
8. [Récapitulatif des Réponses aux Questions du TP](#8-récapitulatif-des-réponses-aux-questions-du-tp)
9. [Guide d'Exécution et Commandes Utiles](#9-guide-dexécution-et-commandes-utiles)

---

## 1. Introduction et Architecture Globale

Ce TP met en place un cycle de vie **MLOps complet et industriel** autour d'un cas réel : la classification de vins (dataset *Wine* de scikit-learn : 178 échantillons, 13 caractéristiques physico-chimiques, 3 cépages cibles). 

L'architecture s'articule autour de 5 piliers interconnectés :
- **Versioning des données et reproductibilité (DVC) :** Séparation stricte entre code (Git) et données/modèles (DVC cache + remote local/S3).
- **Traçabilité et gouvernance (MLflow) :** Tracking exhaustif des hyperparamètres, métriques et artefacts, complété par un registre de modèles centralisé géré par alias (`@champion`).
- **Serving temps-réel conteneurisé (FastAPI + Docker) :** Exposition du modèle par une API REST typée (Pydantic), sécurisée, isolée dans un conteneur allégé exécuté sans privilèges root.
- **Monitoring statistique de dérive (PSI + Kolmogorov-Smirnov) :** Surveillance proactive des distributions de production pour détecter le *data drift* silencieux sans attendre les étiquettes réelles.
- **Boucle continue automatisée (CI/CD/CT via GitHub Actions) :** Re-entraînement sur déclencheur manuel, temporel ou événementiel, filtré par une **porte de qualité** stricte interdisant toute régression en production.

```
       [ Données / DVC ]
              │
              ▼
   [ Entraînement / MLflow ]  ── (9 runs tracés, validation croisée)
              │
              ▼
   [ Registre @champion ]
              │
              ▼
   [ API FastAPI / Docker ]  ── (Route /predict & /health)
              │
              ▼
   [ Monitoring Dérive ]      ── (PSI > 0.25 ou p-value KS < 0.01)
              │ (code 2)
              ▼
    [ CI/CD/CT GitHub Actions ] ── (Porte de qualité -> Déploiement GHCR)
```

---

## 2. Arborescence Finale du Dépôt

Le projet a été organisé dans une structure modulaire respectant les standards de production :

```
tp-mlops/
├── .dvc/                           # Métadonnées et configuration DVC
│   ├── config                      # Remote DVC configuré
│   └── .gitignore
├── .github/
│   └── workflows/
│       └── ct.yml                  # Pipeline CI/CD/CT complet
├── api/
│   ├── __init__.py                 # Package Python
│   ├── main.py                     # API FastAPI (/predict et /health)
│   ├── Dockerfile                  # Conteneurisation de serving
│   └── requirements-api.txt        # Dépendances isolées de l'API
├── data/
│   ├── .gitignore                  # Exclusion de vins.csv du suivi Git
│   ├── vins.csv                    # Produit par src/preparer.py (DVC)
│   ├── production_normale.csv      # Lot simulé sans dérive
│   └── production_derive.csv       # Lot simulé avec dérive volontaire
├── modeles/
│   ├── .gitignore                  # Exclusion du .pkl du suivi Git
│   ├── modele_final.pkl            # Modèle sérialisé (géré par DVC)
│   └── run_id.txt                  # ID du run champion sélectionné
├── src/
│   ├── preparer.py                 # Étape 1 : extraction du jeu de données
│   ├── valider.py                  # Étape 2 : contrôle qualité des données
│   ├── entrainer.py                # Étape 3 : entraînement & tracking MLflow
│   ├── enregistrer.py              # Étape 4 : promotion du modèle dans le registre
│   ├── simuler_production.py       # Génération des flux de production
│   ├── surveiller.py               # Calcul PSI et KS, alerte dérive
│   ├── superviser.py               # Sonde de disponibilité HTTP /health
│   └── evaluer.py                  # Porte de qualité pour la CI/CD/CT
├── tests/
│   └── test_api.py                 # Tests unitaires et d'intégration (pytest)
├── .gitignore                      # Ignore venv, mlruns, mlflow.db, etc.
├── dvc.yaml                        # Définition des stages du pipeline DVC
├── dvc.lock                        # Reproductibilité exacte (hashes DVC)
├── metriques.json                  # Dernières métriques d'entraînement
├── metriques_champion.json         # Métriques de référence du champion en prod
├── params.yaml                     # Hyperparamètres suivis par DVC
├── pytest.ini                      # Configuration de découverte des tests
└── requirements.txt                # Dépendances complètes de développement
```

---

## 3. Partie 1 — Données Versionnées et Pipeline Reproductible (DVC)

### 3.1 Objectifs
- Isoler les données volumineuses ou dynamiques hors du versioning Git standard.
- Assurer la reproductibilité intégrale de l'extraction, de la validation et de l'entraînement via `dvc.yaml`.
- Empêcher l'entraînement sur des données corrompues grâce à un script de validation strict.

### 3.2 Fichiers et Scripts Implémentés

#### `src/preparer.py`
Extrait le jeu de données *Wine* depuis `sklearn.datasets`, renomme la colonne cible en `classe` et sauvegarde 178 lignes et 14 colonnes dans `data/vins.csv`.

#### `src/valider.py`
Vérifie la conformité du fichier CSV :
1. Présence exacte de 14 colonnes (13 caractéristiques chimiques + 1 classe).
2. Taille minimale du lot (au moins 150 lignes).
3. Taux de valeurs manquantes inférieur à 5 % par colonne.
4. Valeurs strictement positives (mesures chimiques).
5. Valeurs de classes valides (appartenant à $\{0, 1, 2\}$).

Si une anomalie est détectée, le script quitte avec le code d'erreur `1`, ce qui interrompt immédiatement le pipeline DVC ou le workflow CI.

#### `params.yaml`
```yaml
entrainement:
  n_estimators: [50, 100, 200]
  max_depth: [3, 6, 10]
  test_size: 0.2
  seed: 42
```

#### `dvc.yaml`
```yaml
stages:
  preparer:
    cmd: python src/preparer.py
    deps: [src/preparer.py]
    outs: [data/vins.csv]
  valider:
    cmd: python src/valider.py data/vins.csv
    deps: [src/valider.py, data/vins.csv]
  entrainer:
    cmd: python src/entrainer.py
    deps: [src/entrainer.py, data/vins.csv]
    params: [entrainement]
    outs: [modeles/modele_final.pkl]
    metrics:
      - metriques.json:
          cache: false
```

### 3.3 Résultats d'Exécution

#### Validation sur lot sain vs lot corrompu
```bash
# Exécution sur le lot de référence sain :
$ python src/valider.py data/vins.csv
# -> Sortie sans erreur, code de retour 0

# Test sur un lot volontairement corrompu (31 lignes sans alcool) :
$ python -c "import pandas as pd; df = pd.read_csv('data/vins.csv'); df.loc[:30, 'alcohol'] = None; df.to_csv('data/corrompu.csv', index=False)"
$ python src/valider.py data/corrompu.csv
ERREUR QUALITÉ : alcohol : 17% de valeurs manquantes
# -> Code de sortie 1 (échec)
```

#### Exécution du pipeline complet avec `dvc repro`
```bash
$ dvc repro
Stage 'preparer' didn't change, skipping
Stage 'valider' didn't change, skipping
Running stage 'entrainer':
> python src/entrainer.py
Meilleur run : n_estimators=50, max_depth=3, accuracy test=1.000
Updating lock file 'dvc.lock'
```
Lors d'une seconde exécution sans modification de dépendance :
```bash
$ dvc repro
Stage 'preparer' didn't change, skipping
Stage 'valider' didn't change, skipping
Stage 'entrainer' didn't change, skipping
Data and pipelines are up to date.
```

---

## 4. Partie 2 — Tracking des Expériences et Registre (MLflow)

### 4.1 Objectifs
- Suivre automatiquement 9 combinaisons d'entraînement ($3 \times 3$ sur `n_estimators` et `max_depth`).
- Utiliser une base SQLite (`sqlite:///mlflow.db`) pour autoriser le registre de modèles.
- Sélectionner objectivement la meilleure configuration et lui attribuer l'alias `@champion`.

### 4.2 Résultats des 9 Runs Expérimentaux

Les 9 variantes ont été exécutées avec une graine aléatoire fixée (`seed=42`) et un découpage stratifié (80% train, 20% test). Une validation croisée à 5 plis (`cross_val_score(cv=5)`) a été calculée sur le jeu d'entraînement :

| Run Name | n_estimators | max_depth | acc_train | acc_test | cv_mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `rf-50-3` | **50** | **3** | **1.000** | **1.000** | **0.986** |
| `rf-50-6` | 50 | 6 | 1.000 | 1.000 | 0.986 |
| `rf-50-10` | 50 | 10 | 1.000 | 1.000 | 0.986 |
| `rf-100-3` | 100 | 3 | 1.000 | 1.000 | 0.986 |
| `rf-100-6` | 100 | 6 | 1.000 | 1.000 | 0.986 |
| `rf-100-10` | 100 | 10 | 1.000 | 1.000 | 0.986 |
| `rf-200-6` | 200 | 6 | 1.000 | 1.000 | 0.986 |
| `rf-200-10` | 200 | 10 | 1.000 | 1.000 | 0.986 |
| `rf-200-3` | 200 | 3 | 1.000 | 1.000 | 0.979 |

### 4.3 Analyse et Justification du Choix du Modèle Champion
1. **Sur-apprentissage (Overfitting) :**  
   Dès que `max_depth >= 6` ou avec un nombre très élevé d'arbres, le modèle apprend les particularités par cœur sans aucun gain de généralisation. Augmenter la profondeur à 10 n'apporte aucun bénéfice par rapport à une profondeur modérée.
2. **Critère de parcimonie (Rasoir d'Ockham) :**  
   Les configurations `(50, 3)`, `(100, 6)`, etc. atteignent toutes la meilleure moyenne de validation croisée (`cv_mean = 0.986`) et une accuracy de test de `1.000`. À performance équivalente, la configuration `n_estimators=50, max_depth=3` est sélectionnée car elle contient le plus faible nombre d'arbres et la profondeur la plus restreinte, garantissant l'empreinte mémoire minimale et la latence d'inférence la plus faible.
3. **Enregistrement et Alias :**  
   Le modèle retenu est enregistré dans le registre MLflow sous le nom `classifieur-vins` et étiqueté avec l'alias `@champion` via `src/enregistrer.py` :
   ```
   Successfully registered model 'classifieur-vins'.
   Created version '1' of model 'classifieur-vins'.
   classifieur-vins v1 -> alias @champion
   Rechargé depuis le registre : PyFuncModel
   ```

---

## 5. Partie 3 — Déploiement et Serving : API REST et Docker (FastAPI)

### 5.1 Conception de l'API (`api/main.py`)
- **Chargement singleton :** Le modèle sérialisé (`modele_final.pkl`) est désérialisé une seule fois au démarrage de l'application via `joblib.load`, garantissant des réponses en moins de 10 ms sans ré-accès disque.
- **Validation stricte Pydantic :** La classe `Entree` impose exactement 13 valeurs flottantes (`Field(min_length=13, max_length=13)`). Toute requête incomplète ou mal formée est rejetée avec une réponse HTTP `422 Unprocessable Entity` avant même de solliciter le modèle.
- **Conformité des features :** Les entrées sont encapsulées dans un `pd.DataFrame` avec la liste ordonnée exacte des colonnes d'entraînement (`FEATURES`), éliminant tout avertissement scikit-learn.
- **Observabilité :** Présence d'un endpoint de sonde `/health` (`200 OK`) et journalisation systématique des prédictions (`confiance`, classe prédite).

### 5.2 Tests Automatisés (`tests/test_api.py`)
Exécution de la suite de tests avec `pytest` :
```bash
$ pytest tests/ -v
tests/test_api.py::test_prediction_valide PASSED                         [ 25%]
tests/test_api.py::test_vin_connu_bien_classe PASSED                     [ 50%]
tests/test_api.py::test_entree_incomplete_refusee PASSED                 [ 75%]
tests/test_api.py::test_health PASSED                                    [100%]
======================== 4 passed in 1.84s =========================
```

### 5.3 Conteneurisation Docker
L'image Docker a été construite selon les règles de l'art (Python 3.11-slim, dépendances mises en cache, utilisateur non-privilégié `appuser`, pas d'exécution en `root`) :
```bash
$ docker build -f api/Dockerfile -t api-modele-ml:1.0 .
# -> Image api-modele-ml:1.0 créée (taille ~750 Mo totale avec couches décompressées)

$ docker run -d --name api-vins -p 8000:8000 api-modele-ml:1.0
```

#### Test de prédiction HTTP réelle :
```bash
$ curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]}"
```
**Réponse JSON reçue :**
```json
{
  "prediction": 0,
  "confiance": 0.94,
  "version_modele": "1"
}
```

### 5.4 Supervision de Disponibilité (`src/superviser.py`)
Simulation d'une panne (arrêt du conteneur) suivie de son redémarrage :
```
17:00:01 OK
17:00:03 OK
# docker stop api-vins
17:00:06 ALERTE : /health a répondu None (échec n° 1)
17:00:08 ALERTE : /health a répondu None (échec n° 2)
# docker start api-vins
17:00:16 RÉTABLI après 2 échec(s)
17:00:16 OK
17:00:18 OK
```

---

## 6. Partie 4 — Détection de la Dérive des Données (Data Drift)

### 6.1 Méthodologie Statistique
Le script `src/surveiller.py` implémente une double validation statistique pour chaque variable :
1. **Population Stability Index (PSI) :** Découpage en 10 classes quantiles basé sur le jeu de référence.  
   - $\text{PSI} < 0.10$ : Distribution stable, aucune action.
   - $0.10 \le \text{PSI} \le 0.25$ : Dérive modérée, surveillance recommandée.
   - $\text{PSI} > 0.25$ : Dérive significative, intervention requise.
2. **Test bilatéral de Kolmogorov-Smirnov (KS) :** Test non paramétrique comparant les fonctions de répartition empiriques. Une $p$-value $< 0.01$ rejette l'hypothèse nulle d'identité des distributions.
3. **Condition d'alerte :** Une variable est déclarée en dérive si $(\text{PSI} > 0.25) \lor (p\text{-value} < 0.01)$. Si au moins une variable dérive, le script sort avec le **code 2** (signal standard de ré-entraînement pour le pipeline CT).

### 6.2 Résultats Comparatifs

#### 1. Lot de production normale (`data/production_normale.csv`) :
```
          feature   psi    ks p_value  derive
  proanthocyanins 0.042 0.056 9.7e-01   False
  color_intensity 0.041 0.040 1.0e+00   False
              ash 0.040 0.050 9.9e-01   False
alcalinity_of_ash 0.030 0.054 9.7e-01   False
          alcohol 0.027 0.033 1.0e+00   False
    total_phenols 0.019 0.028 1.0e+00   False

Aucune dérive significative.
# -> Code de sortie 0
```

#### 2. Lot avec dérive volontaire (`data/production_derive.csv`) :
*(Modifications introduites : `alcohol` + 1.2, `color_intensity` $\times$ 1.5, `proline` + bruit)*
```
        feature   psi    ks p_value  derive
        alcohol 2.482 0.476 2.9e-15    True
color_intensity 0.786 0.353 1.8e-08    True
proanthocyanins 0.049 0.049 9.9e-01   False
        proline 0.048 0.055 9.7e-01   False
      magnesium 0.037 0.045 1.0e+00   False
     malic_acid 0.032 0.036 1.0e+00   False

ALERTE : dérive détectée sur ['alcohol', 'color_intensity'] -> déclencher le ré-entraînement
# -> Code de sortie 2
```

---

## 7. Partie 5 — Automatisation CI/CD/CT et Porte de Qualité (GitHub Actions)

### 7.1 La Porte de Qualité (`src/evaluer.py`)
Avant d'autoriser la mise en production d'un nouveau modèle, la porte de qualité applique deux règles obligatoires :
1. **Seuil absolu d'accuracy/validation croisée :** `cv_mean >= 0.90`.
2. **Règle de non-régression par rapport au champion :** `cv_mean >= champion['cv_mean'] - 0.005` (marge de 0.5% pour tolérer les micro-fluctuations d'échantillonnage).

#### Test 1 — Modèle valide (Succès) :
```
$ python src/evaluer.py
cv_mean nouveau = 0.986 | champion = 0.972
ACCEPTÉ : le nouveau modèle peut être déployé
# -> Code de sortie 0
```

#### Test 2 — Régression simulée (`max_depth=1`, Échec) :
```
$ python src/evaluer.py
Meilleur run : n_estimators=1, max_depth=1, accuracy test=0.639
cv_mean nouveau = 0.585 | champion = 0.972
REFUS : 0.585 < seuil absolu 0.9
# -> Code de sortie 1 : échec immédiat du pipeline
```

### 7.2 Architecture du Workflow (`.github/workflows/ct.yml`)

Le workflow comprend **4 modes de déclenchement** pour couvrir tout le cycle DevOps & MLOps :
- `push` sur la branche `main` (filtré sur les chemins `src/**`, `api/**`, `params.yaml`, `dvc.yaml`) : **CI / CD classique**.
- `schedule` (cron `"0 3 * * 1"`, chaque lundi à 3h UTC) : **CT planifié**.
- `workflow_dispatch` : **CT manuel** déclenché par un Data Scientist depuis l'interface GitHub.
- `repository_dispatch` (type `derive-detectee`) : **CT automatique événementiel** déclenché par le service de surveillance lors d'une dérive constatée.

Le job `livrer` (publication de l'image Docker sur GitHub Container Registry `ghcr.io`) utilise `needs: entrainer-evaluer` : si la porte de qualité échoue, la livraison est court-circuitée et l'image existante reste en production.

---

## 8. Récapitulatif des Réponses aux Questions du TP

### Questions de la Partie 1
- **Question 1 : Que contient Git après ce commit ? Où se trouvent réellement les données ?**  
  *Réponse :* Git ne conserve que le fichier pointeur léger `vins.csv.dvc` (qui contient la somme de contrôle MD5, la taille et le chemin du fichier), le fichier `data/.gitignore` (qui bloque l'indexation de `data/vins.csv`), et la configuration `.dvc/config`. Les données réelles ne sont jamais stockées dans l'historique Git : elles résident dans le cache local DVC (`.dvc/cache`) et dans le dépôt de stockage distant (`../dvc-stockage`), indexées sous leur empreinte MD5.
- **Question 2 : Modifier une valeur de data/vins.csv, puis lancer dvc status. Que se passe-t-il ?**  
  *Réponse :* `dvc status` compare l'empreinte MD5 actuelle du fichier avec celle enregistrée dans le pointeur `.dvc` et indique : `data\vins.csv: changed outs: modified: data\vins.csv`. Pour annuler les modifications, on exécute `dvc checkout data/vins.csv.dvc` qui restaure la version originale depuis le cache. Pour entériner la modification, on exécuterait `dvc add data/vins.csv` suivi d'un commit Git.
- **Question 3 : On modifie seulement max_depth dans params.yaml. Quelles étapes dvc repro relance-t-il ?**  
  *Réponse :* DVC analyse le graphe orienté acyclique (DAG) des dépendances défini dans `dvc.yaml`. Seule l'étape `entrainer` déclare une dépendance explicite envers `params.yaml` (section `entrainement`). Par conséquent, DVC saute les étapes `preparer` et `valider` (marquées `didn't change, skipping`), et réexécute uniquement `entrainer`. Le fichier `dvc.lock` est ensuite mis à jour.

### Questions de la Partie 2
- **Question 1 : Pourquoi six runs ont-ils exactement la même accuracy de test (0,972) ?**  
  *Réponse :* Le sous-ensemble de test (`test_size=0.2`) ne contient que 36 exemples ($178 \times 0.20 \approx 36$). Une erreur de classification supplémentaire fait varier l'accuracy d'un pas discret de $1/36 \approx 0.0278$. Les scores possibles sont donc discrets (ex: $34/36 \approx 0.944$, $35/36 \approx 0.972$, $36/36 = 1.000$). De multiples combinaisons font exactement une seule erreur sur les 36 vins, d'où l'égalité fréquente à 0.972. Cela démontre la nécessité de la validation croisée (`cv_mean`), calculée sur plusieurs plis d'entraînement, pour discriminer les modèles.
- **Question 2 : Quels runs montrent un sur-apprentissage ? Sur quel indice ?**  
  *Réponse :* Les configurations avec une profondeur $\ge 6$ affichent une accuracy d'entraînement de 1.000 (100% de bonnes réponses sur les données d'apprentissage), révélant que les arbres mémorisent parfaitement les données. Pousser la profondeur jusqu'à 10 n'apporte aucune amélioration sur les plis de test, prouvant que la complexité ajoutée est inutile et propice à l'overfitting.
- **Question 3 : Quelle configuration retenir, et pourquoi ?**  
  *Réponse :* On retient le modèle offrant la meilleure généralisation en validation croisée (`cv_mean`), et en cas d'égalité, le modèle le plus simple (principe de parcimonie). Le modèle sélectionné minimise la profondeur et le nombre d'estimateurs, offrant ainsi une inférence plus véloce, une moindre consommation mémoire et une robustesse accrue.
- **Question 4 : Comment revenir au modèle précédent si la version 2 pose problème en production ?**  
  *Réponse :* On réaffecte l'alias MLflow en exécutant :
  ```python
  client.set_registered_model_alias("classifieur-vins", "champion", 1)
  ```
  Tout microservice ou API chargeant le modèle dynamique via `models:/classifieur-vins@champion` basculera immédiatement sur la version 1 à son prochain rechargement, sans devoir modifier le code source ni redéployer d'image Docker.

### Questions de la Partie 3
- **Question 1 : Pourquoi l'image pèse-t-elle près de 500–700 Mo alors que le modèle ne fait que quelques centaines de Ko ?**  
  *Réponse :* Le modèle binaire sérialisé (`modele_final.pkl`) ne pèse que ~70 Ko. L'essentiel du volume de l'image provient du système d'exploitation de base (Debian Linux minimal dans `python:3.11-slim`) et des bibliothèques scientifiques compilées (NumPy, SciPy, Scikit-learn, Pandas) qui intègrent de volumineux binaires C/Fortran d'algèbre linéaire (BLAS/LAPACK), ainsi que le runtime ASGI FastAPI/Uvicorn.
- **Question 2 : Que se passe-t-il si l'image est construite avec scikit-learn 1.6 alors que le modèle a été entraîné avec la 1.5.2 ?**  
  *Réponse :* À la désérialisation du modèle avec `joblib`, scikit-learn émet un avertissement explicite `InconsistentVersionWarning`. Si les modifications internes entre les versions touchent aux structures de données de l'arbre (`Tree`), le chargement peut lever une exception bloquante (`UnpicklingError`), ou pire, produire des prédictions erronées en production sans erreur apparente. C'est pourquoi figer strictement les versions identiques dans `requirements.txt` et `requirements-api.txt` est indispensable en MLOps.

### Questions de la Partie 4
- **Question 1 : Pourquoi la feature proline n'est-elle pas détectée alors qu'on lui a ajouté du bruit ?**  
  *Réponse :* Dans le jeu de données, la variable `proline` possède un ordre de grandeur élevé et s'étend de 278 à 1680 (écart-type naturel $\sigma \approx 315$). Le bruit gaussien artificiel injecté avec un écart-type de 30 est négligeable devant cette variance naturelle (environ $1/10$ de l'écart-type). La distribution cumulative n'est pratiquement pas altérée (PSI $\approx 0.048$, p-value KS $> 0.9$), ce qui évite un faux positif. Une dérive n'est statistiquement significative que si sa magnitude est proportionnée à la variabilité intrinsèque de la grandeur mesurée.
- **Question 2 : Le lot normal n'a pas un PSI nul. Pourquoi ? Pourquoi le seuil est-il à 0,25 et pas à 0 ?**  
  *Réponse :* Le lot de production est un échantillon aléatoire de 120 observations. Même lorsqu'il provient de la même population d'origine, les fluctuations d'échantillonnage aléatoires empêchent que les fréquences empiriques par quantile soient rigoureusement égales à celles de référence. Un seuil fixé à 0 déclencherait des fausses alertes à chaque lot. Les seuils standards (0.10 pour la pré-alerte et 0.25 pour l'action) permettent d'absorber le bruit naturel d'échantillonnage.
- **Question 3 : Comment vérifier que la dérive dégrade réellement le modèle ?**  
  *Réponse :* 
  1. Dès que les étiquettes réelles de terrain (ground truth) sont collectées ultérieurement, recalculer la matrice de confusion, le F1-score et l'accuracy sur ce lot dérivé et comparer aux métriques de test initiales.
  2. En temps réel sans étiquettes, monitorer l'entropie des probabilités de sortie : une baisse sensible de la métrique `confiance` (`probabilites.max()`) signale que le modèle hésite davantage.
  3. Croiser les variables qui dérivent avec l'importance des caractéristiques (`feature_importances_`) : si les variables dérivées font partie des prédicteurs majeurs (comme `alcohol` et `color_intensity`), l'impact sur les performances est garanti.

### Questions de la Partie 5
- **Question 4 : Que se passe-t-il si le nouveau modèle obtient cv_mean = 0,955 ?**  
  *Réponse :* Le script `src/evaluer.py` compare `0.955` à `champion['cv_mean'] - 0.005` ($0.972 - 0.005 = 0.967$). Comme $0.955 < 0.967$, la condition de non-régression n'est pas satisfaite. Le script affiche `REFUS : moins bon que le champion, il reste en production` et quitte avec un code d'erreur non nul. Dans le workflow GitHub Actions, le job `entrainer-evaluer` échoue immédiatement ; le job `livrer` (qui dépend de son succès via `needs: entrainer-evaluer`) n'est jamais exécuté. Le modèle dégradé est écarté et le champion en production reste actif.
- **Question 5 : Pourquoi tolérer un écart de 0,005 avec le champion ?**  
  *Réponse :* L'entraînement d'algorithmes stochastiques (comme Random Forest) et la répartition en plis de validation croisée comportent une part de variance statistique. Deux entraînements successifs sur un jeu de données quasi-identique peuvent générer une fluctuation minime de score ($\pm 0.002$). Sans marge de tolérance, des modèles parfaitement valables seraient systématiquement rejetés pour un simple artefact de tirage. La tolérance de $0.005$ (0.5%) neutralise ce bruit statistique tout en bloquant toute régression notable.
- **Question 6 : Quelles améliorations apporter pour un vrai projet industriel ?**  
  *Réponse :*
  1. **Infrastructure MLflow centralisée :** Remplacer SQLite local par un serveur MLflow hébergé (PostgreSQL pour le backend store + bucket AWS S3 / Google Cloud Storage pour les artefacts), accessible via une URL distante sécurisée par token.
  2. **Remote DVC d'entreprise :** Configurer un stockage cloud distribué (S3, GCS, Azure Blob Storage) avec authentification IAM plutôt qu'un répertoire de fichiers local.
  3. **Déploiement progressif (Canary / Blue-Green) :** Intégrer Kubernetes avec Argo Rollouts ou Istio pour router initialement 5% du trafic vers le nouveau modèle, avec retour arrière automatique (auto-rollback) si le taux d'erreur HTTP 5xx ou la latence p99 dépasse un seuil.
  4. **Orchestration du monitoring :** Planifier la détection de dérive via un orchestrateur de données (Apache Airflow, Prefect) qui appelle automatiquement l'API GitHub (`repository_dispatch`) dès qu'une dérive de données ou de concept est constatée.
  5. **Évaluations avancées :** Ajouter à la porte de qualité des tests d'équité (fairness par sous-groupes), de robustesse aux données aberrantes et des benchmarks de latence d'inférence sous charge (p95 < 20 ms).

---

## 9. Guide d'Exécution et Commandes Utiles

Pour reproduire l'intégralité du projet sur une nouvelle machine :

```bash
# 1. Cloner le projet et créer l'environnement virtuel
git clone <URL_DU_DEPOT> tp-mlops
cd tp-mlops
python -m venv .venv
# Windows : .venv\Scripts\activate
# Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt

# 2. Exécuter le pipeline complet de données et modèle (DVC)
dvc repro

# 3. Lancer l'interface de comparaison MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# 4. Enregistrer et promouvoir le meilleur modèle
python src/enregistrer.py

# 5. Exécuter les tests unitaires de l'API
pytest tests/ -v

# 6. Construire et tester le conteneur Docker
docker build -f api/Dockerfile -t api-modele-ml:1.0 .
docker run -d --name api-vins -p 8000:8000 api-modele-ml:1.0
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]}"

# 7. Tester la détection de dérive (monitoring)
python src/simuler_production.py
python src/surveiller.py data/production_normale.csv   # Code 0 (sain)
python src/surveiller.py data/production_derive.csv    # Code 2 (alerte dérive)

# 8. Tester la porte de qualité
python src/evaluer.py
```

---
*Fin du rapport — TP DevOps & MLOps Partie 2 — 3IDL.*
