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
import requests
import langchain_core
from bs4 import BeautifulSoup
import urllib.parse
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # You can choose any supported model


def google_search(query):
    # Prepare the query for the URL
    url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
    
    # Set headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    
    # Send a GET request to the Google search page
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find search result elements
        results = soup.find_all('h3')  # Google uses <h3> for titles in search results
        links = soup.find_all('a')  # To get URLs
        
        web_results = []  # To store search results
        
        # Extract titles and links and store them in web_results
        for title, link in zip(results, links):
            result = {
                'title': title.get_text(),
                'link': link.get('href')
            }
            web_results.append(result)

        # Create a Document object with the web results
        web_results_doc = Document(page_content=str(web_results))
        return web_results_doc
    else:
        print("Error fetching results:", response.status_code)
class DocumentVectors():
    def __init__(self,document:Document,vector:list[float]):
        
        self.page_content = document.page_content
        self.vector = vector

class RAGPipeline():
    def __init__(self,model_name,context):
        self.prompt = hub.pull("rlm/rag-prompt")
        self.model = ChatGroq(temperature=0,
                              model_name=model_name,
                              api_key="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")
        self.output_parser = StrOutputParser()
        self.update_context(context) if context else None
    def update_context(self,context):
        assert(len(context)>0 and type(context[0]) == Document)
        self._context = context
    def invoke(self,data):
        
        if isinstance(data, str): # build prompt
            prompt_config = {
                "question": data,
                "context": self._context
            }                
            prompt = self.prompt.invoke(prompt_config)
        elif isinstance(data,langchain_core.prompt_values.PromptValue):
            prompt = data
        else:
            print(type(data))
            raise "Invalid data type"
        llm_output = self.model.invoke(prompt)
        output= self.output_parser.invoke(llm_output)
        
        return output
    async def invokeAsync(self,data):
        
        if isinstance(data, str): # build prompt
            prompt_config = {
                "question": data,
                "context": self._context
            }                
            prompt = self.prompt.invoke(prompt_config)
        elif isinstance(data,langchain_core.prompt_values.PromptValue):
            prompt = data
        else:
            raise "Invalid data type"
        llm_output =  self.model.invoke(prompt)
        output= self.output_parser.invoke(llm_output)
        
        return output
        
    
    @staticmethod
    def retrieve_docs(path,query,count=4,with_vectors=False):
        
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = FAISS.load_local(path, embedding_model,"embeddings",allow_dangerous_deserialization=True)
        
        
        
        retrieved_docs = vector_store.similarity_search(query,count)
        if(with_vectors):
            retrieved_docs = [DocumentVectors(doc,embedding_model.embed_query(doc.page_content)) for doc in retrieved_docs]
        return retrieved_docs



query_prompt = "How Transformers work?"
retrieved_docs = RAGPipeline.retrieve_docs("./vectorestore/",query_prompt)

 
rag = RAGPipeline("llama3-8b-8192",retrieved_docs)
rag.invoke(query_prompt)

