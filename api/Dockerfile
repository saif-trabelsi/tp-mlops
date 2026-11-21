# Image de l'API : modèle + code + bibliothèques figées (TP 5 du cours, adapté à l'arborescence)
FROM python:3.11-slim
WORKDIR /app

# 1) Dépendances d'abord : couche mise en cache tant que la liste ne change pas
COPY api/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 2) Le modèle entraîné et le code de l'API
COPY modeles/modele_final.pkl api/main.py ./

# 3) Exécution sans privilèges
RUN useradd --create-home appuser
USER appuser

ENV MODEL_PATH=modele_final.pkl MODEL_VERSION=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
