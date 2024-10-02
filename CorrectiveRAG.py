# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:17:16 2024

@author: mitran
"""


from langchain.prompts import PromptTemplate
from langchain_core.documents.base import Document
import requests
from bs4 import BeautifulSoup
import urllib.parse
from RAGPipeline import RAGPipeline

def get_grade_prompt():
    return PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n
    Here is the retrieved document: \n\n {context} \n\n
    Here is the user question: {question} \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\n
    provide string 'yes' or'no' only""",
    input_variables=["context", "question"],
)


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


class CorrectiveRAG(RAGPipeline):
    def __init__(self, model_name, context,count=5):
        super().__init__(model_name, context)
        
        self.grade_prompt = get_grade_prompt()
        self.count = count
    def invoke(self,question):
        updated_context = self.grade_documents(question)
        self.update_context(updated_context)
        op = super().invoke(question)
        print(question,': \n',op)

        return op
    def grade_documents(self,question):
        top_docs = []
        for i,docs in enumerate(self._context) :
            prompt_config = {
                "context":docs,
                "question":question
                }
            decision = super().invoke(question,self.grade_prompt.invoke(prompt_config))
            if(decision=="yes"):
                top_docs.append(docs)
        print("matched "+str(len(top_docs))+" documents")
        if(len(top_docs)>=self.count):
            return top_docs[:self.count]
        else:
            print("searching web")
            top_docs.append(google_search(question))
            return top_docs
                
        
query_prompt = "why we use Transformers?"
retrieved_docs = RAGPipeline.retrieve_docs("./vectorestore/",query_prompt,10)

 
rag = CorrectiveRAG("llama3-8b-8192",retrieved_docs)

rag.invoke("what is transformer")
rag.invoke("what is diffusion models")





  
    
        
