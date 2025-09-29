# import os
# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# DB_USER = os.getenv("DB_USER", "admin")
# DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
# DB_HOST = os.getenv("DB_HOST", "mysql-db")
# DB_NAME = os.getenv("DB_NAME", "admin")
# DB_PORT = os.getenv("DB_PORT", "3306")

# DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# engine = create_engine(DATABASE_URL, pool_pre_ping=True)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()


from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Adapter avec tes infos MySQL
DATABASE_URL = "mysql+pymysql://root:@localhost:3306/vehicule_db"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
