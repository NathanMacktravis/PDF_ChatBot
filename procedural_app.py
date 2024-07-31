from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os 
import shutil
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from constants import CHROMA_SETTINGS

# Configuration du périphérique
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = "LaMini-T5-738M"

# Chargement du tokenizer et du modèle
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, 
    torch_dtype = torch.float32
)

# Fonction pour créer le pipeline de génération de texte
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.95,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

# Fonction pour créer l'objet RetrievalQA
def qa_llm():
    # Supprimer l'index existant pour éviter les problèmes de corruption
    if os.path.exists("db"):
        shutil.rmtree("db")

    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function = embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa

# Fonction pour traiter la réponse à une question
def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

# Fonction principale pour interagir avec l'utilisateur
def main():
    print("Question & Answer app PDF 🦜📄")
    print("This is a power Generative Question & Answer App that responds to questions about your PDF file.")
    
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = process_answer(question)
        print(f"Your question: {question}")
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
