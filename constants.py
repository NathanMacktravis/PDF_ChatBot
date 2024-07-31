import os 
import chromadb
from chromadb.config import Settings 

# Configuration des paramètres pour ChromaDB

#client = chromadb.PersistentClient(path="db", settings=Settings(anonymized_telemetry=False))

CHROMA_SETTINGS = Settings(
    chroma_db_impl = 'duckdb+parquet',  # Spécifie l'implémentation de la base de données à utiliser, ici 'duckdb+parquet' indique que DuckDB est utilisé avec Parquet comme format de stockage.
    persist_directory = 'db',  # Définit le répertoire où les données persistantes de la base de données seront stockées
    anonymized_telemetry = False  # Indique si la télémétrie (collecte de données d'utilisation) doit être anonymisée
)