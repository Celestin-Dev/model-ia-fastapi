from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


DB_USER= "root"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "vehicule_db"

# URL de connexion MySQL (adapter avec tes infos)
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Créer l'engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Créer une session locale
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base pour les modèles
Base = declarative_base()
