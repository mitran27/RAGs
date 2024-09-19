# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:35:58 2024

@author: mitran
"""

import os
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # Replace with your chosen vector store
import uuid
import time

COLLECTION_NAME = "rag"
DB_PATH = "./chromaDB"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # You can choose any supported model
def load_docs(urls:list):
    from langchain_community.document_loaders import WebBaseLoader
    
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
    )
    doc_split = text_splitter.split_documents(docs_list)
    
    return doc_split        
 

    
def process_docs(docs):
    return {
        "texts" : [doc.page_content for doc in docs],
        #"metadatas" : [doc.metadata for doc in docs],
        "ids":[str(uuid.uuid4()) for _ in docs]
        }
assert len(os.getenv("OPENAI_API_KEY")) > 0

split_docs = load_docs([
    "https://jalammar.github.io/illustrated-transformer/",
    "https://jalammar.github.io/illustrated-bert/",
    "https://jalammar.github.io/illustrated-retrieval-transformer/",
    ])


embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vector_store = FAISS.from_documents(split_docs, embedding_model)

vector_store.save_local("./vectorestore/","embeddings")
del(vector_store)