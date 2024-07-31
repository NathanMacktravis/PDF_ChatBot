### PDF Chatbot App

Ce projet est une application de **Question & Réponse Générative** (RAG) qui permet aux utilisateurs de poser des questions à partir de documents PDF et d'obtenir des réponses générées automatiquement par un modèle de langage avancé. L'application combine des techniques de traitement du langage naturel (NLP) et de récupération d'informations pour fournir des réponses précises et pertinentes basées sur le contenu des PDF.

### Technologies Utilisées

- **Transformers (Hugging Face)** : Utilisé dans ce projet pour charger et exécuter le modèle de langage pré-entraîné *LaMini-T5-738M*.

- **PyTorch** : Framework utilisé pour manipuler les modèles de deep learninget pour gérer les opérations sur les GPU afin d'accélérer l'inférence du modèle.

- **LangChain** : Utilisé pour orchestrer les différentes étapes du pipeline de traitement de texte, y compris l'extraction de réponses pertinentes à partir des documents PDF et la génération de réponses. 

- **Chroma** : Base de données vectorielle utilisée pour le stockage et l'indexation afin de  rechercher efficacement les segments de texte issus des PDF.

- **Sentence Transformers** : Utilisé pour créer des embeddings vectoriels des segments de texte, facilitant ainsi la recherche sémantique dans les documents.

- **Streamlit** :  Interface utilisateur interactive utilisée au départ pour le développement et le test de l'application. 
Cependant, le projet dispose de deux versions : 
    - procedural_app.py : pour une utilisation directe via la console.
    - streamlit_app.py : pour une utilisation avec l'interface streeamlit

- **Python** : Le langage de programmation principal utilisé pour intégrer toutes les bibliothèques et technologies mentionnées ci-dessus, en orchestrant l'ensemble du flux de travail du projet.


