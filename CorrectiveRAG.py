# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:17:16 2024

@author: mitran
"""

from langchain.vectorstores import FAISS  # Replace with your chosen vector store
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # You can choose any supported model

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_store = FAISS.load_local("./vectorestore/", embedding_model,"embeddings",allow_dangerous_deserialization=True)


query = "what is Transformer ?"
retrieved_docs = vector_store.similarity_search(query)
print(retrieved_docs)

  
    
        
