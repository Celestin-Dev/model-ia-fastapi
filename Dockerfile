FROM python:3.13-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copier tout le code
COPY . .

# Commande de démarrage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
