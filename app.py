from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import streamlit 
import torch
import base64
import textwrap
import streamlit as st 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chain import RetrievalQA
from langchain.llms import HugginFacePipeline
from constants import CHROMA_SETTINGS

checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, 
    device_map = "auto",
    toch_dtype = torch.float32
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
    local_llm = HugginFacePipeline(pipeline=pipe)
    return local_llm
