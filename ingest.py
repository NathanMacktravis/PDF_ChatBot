# Importation des bibliothèques nécessaires pour le traitement des documents PDF, le découpage de texte, la création d'embeddings et la gestion de la base de données vectorielle
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
import os 
from constants import CHROMA_SETTINGS


# Répertoire où les données vectorielles seront persistées
persist_directory = "db"

def main():
    # Parcours récursif du répertoire "docs" pour trouver les fichiers PDF
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                # Chargement du fichier PDF à l'aide de PyPDFLoader
                loader = PyPDFLoader(os.path.join(root, file))
                
    # Chargement des documents PDF
    documents = loader.load()
    
    # Initialisation du découpeur de texte pour diviser les documents en morceaux de texte plus petits
    print("------splitting into chunks-------")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    # Création des embeddings à partir des morceaux de texte
    print("------Loading sentence transformers model------")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Création de la base de données vectorielle et stockage des embeddings
    print(f"------Creating embeddings. May take some minutes...------")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    
    # Persistance de la base de données vectorielle
    db.persist()
    
    # Libération de la mémoire en supprimant la référence à la base de données
    db = None 


if __name__ == "__main__":
    main()

     
    