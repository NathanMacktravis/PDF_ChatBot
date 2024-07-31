## Importation des bibliothèques nécessaires
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Pour le chargement du tokenizer et du modèle de transformation
from transformers import pipeline  # Pour créer un pipeline de génération de texte
import torch  # Pour gérer les tensors et les modèles sur CPU ou GPU
from langchain.embeddings import SentenceTransformerEmbeddings  # Pour générer des embeddings de texte
from langchain.vectorstores import Chroma  # Pour créer une base de données vectorielle
from langchain.chains import RetrievalQA  # Pour créer une chaîne de question-réponse basée sur la récupération
from langchain.llms import HuggingFacePipeline  # Pour créer un modèle de LLM personnalisé à partir de pipelines HuggingFace
from constants import CHROMA_SETTINGS  # Importation des paramètres spécifiques à Chroma (fichier constants.py)
import os 
import shutil  # Pour manipuler les fichiers et les dossiers

## Configuration du périphérique (GPU si disponible, sinon CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = "LaMini-T5-738M"  # Modèle pré-entraîné choisi pour la tâche

## Chargement du tokenizer et du modèle depuis les fichiers pré-entraînés
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, 
    torch_dtype = torch.float32  # Spécification du type de données utilisé par le modèle
)

## Fonction pour créer le pipeline de génération de texte avec le modèle HuggingFace
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',  # Type de tâche : génération de texte à partir de texte
        model = base_model,  # Modèle à utiliser pour la génération
        tokenizer = tokenizer,  # Tokenizer associé au modèle
        max_length = 512,  # Longueur maximale des séquences générées
        do_sample = True,  # Autorisation de l'échantillonnage lors de la génération (génération plus variée)
        temperature = 0.3,  # Contrôle de la créativité (valeur plus basse = plus conservateur)
        top_p= 0.95,  # Seuil pour la méthode de décodage top-p (ou nucleus sampling)
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)  # Enveloppement du pipeline dans un objet LangChain
    return local_llm

## Fonction pour créer un objet de question-réponse basé sur la récupération (RetrievalQA)
def qa_llm():
    # Supprimer l'index existant pour éviter les problèmes de corruption
    if os.path.exists("db"):
        shutil.rmtree("db")  # Supprime le dossier "db" s'il existe

    llm = llm_pipeline()  # Création du pipeline de génération de texte
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  # Génération d'embeddings à partir du modèle SentenceTransformer
    db = Chroma(persist_directory="db", embedding_function = embeddings, client_settings=CHROMA_SETTINGS)  # Création de la base de données vectorielle avec persistance des données
    retriever = db.as_retriever()  # Conversion de la base de données en un objet de récupération de documents
    qa = RetrievalQA.from_chain_type(
        llm = llm,  # Utilisation du LLM pour générer des réponses
        chain_type = "stuff",  # Type de chaîne utilisé (ici, une chaîne simple)
        retriever = retriever,  # Système de récupération pour trouver les documents pertinents
        return_source_documents=True  # Retourne les documents sources avec les réponses
    )
    return qa

## Fonction pour traiter la réponse à une question donnée
def process_answer(instruction):
    qa = qa_llm()  # Initialisation de l'objet de question-réponse
    generated_text = qa(instruction)  # Génération de la réponse à partir de l'instruction (question)
    answer = generated_text['result']  # Extraction de la réponse générée
    return answer

# Fonction principale pour interagir avec l'utilisateur via la console
def main():
    print("Question & Answer app PDF 🦜📄")  # Titre de l'application
    print("This is a power Generative Question & Answer App that responds to questions about your PDF file.")  # Description de l'application
    
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")  # Entrée de la question par l'utilisateur
        if question.lower() == 'exit':  # Condition pour quitter la boucle
            break
        answer = process_answer(question)  # Traitement de la question et obtention de la réponse
        print(f"Your question: {question}")  # Affichage de la question posée
        print(f"Answer: {answer}")  # Affichage de la réponse obtenue


if __name__ == "__main__":
    main()  