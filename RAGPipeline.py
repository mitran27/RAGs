# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:39:33 2024

@author: mitran
"""

from langchain.vectorstores import FAISS  # Replace with your chosen vector store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.documents.base import Document



EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # You can choose any supported model





class RAGPipeline():
    def __init__(self,model_name,context):
        self.prompt = hub.pull("rlm/rag-prompt")
        self.model = ChatGroq(temperature=0,
                              model_name=model_name,
                              api_key="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")
        self.output_parser = StrOutputParser()
        self.update_context(context)
    def update_context(self,context):
        assert(len(context)>0 and type(context[0]) == Document)
        self._context = context
    def invoke(self,question,prompt=None):
        if prompt is None:
            input_config = {
                "question": question,
                "context": self._context
            }
                
            prompt = self.prompt.invoke(input_config)
        llm_output = self.model.invoke(prompt)
        output= self.output_parser.invoke(llm_output)
        
        return output
    
    @staticmethod
    def retrieve_docs(path,query,count=4):
        
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = FAISS.load_local(path, embedding_model,"embeddings",allow_dangerous_deserialization=True)
        
        
        retrieved_docs = vector_store.similarity_search(query,count)
        return retrieved_docs



query_prompt = "How Transformers work?"
retrieved_docs = RAGPipeline.retrieve_docs("./vectorestore/",query_prompt)

 
rag = RAGPipeline("llama3-8b-8192",retrieved_docs)
rag.invoke(query_prompt)

