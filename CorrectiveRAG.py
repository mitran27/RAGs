# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:17:16 2024

@author: mitran
"""


from RAGPipeline import RAGPipeline, google_search

from Prompts import GRADING_PROMPT



class CorrectiveRAG(RAGPipeline):
    def __init__(self, model_name, context,count=5):
        super().__init__(model_name, context)
        
        self.grade_prompt = GRADING_PROMPT
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





  
    
        
