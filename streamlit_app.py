from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import streamlit 
import torch
import base64
import textwrap
import streamlit as st 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from constants import CHROMA_SETTINGS
import os 
import shutil


checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, 
    device_map = "auto",
    torch_dtype = torch.float32
)

@st.cache_resource #Pour éviter de charger le modèle à chaque fois durant l'exécution 
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


@st.cache_resource
def qa_llm():
    # Supprimer l'index existant pour éviter les problèmes de corruption
    """if os.path.exists("db"):
        shutil.rmtree("db")"""
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



def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer



def main(): 
    st.title("Question & Answer app PDF 🦜📄")
    with st.expander("About the app") : 
        st.markdown("""
                    This is a power Generative Question & Answer App
                    that responds to questions about your PDF file.
                    """
                    )
        
    question = st.test_area("Enter your question")
    if st.button("Search"):
        st.info("Your question : "+question)
        st.info("Your answer")
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)


if __name__ == '___main__':
    main()
