# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:04:04 2024

@author: mitran
"""



from RAGPipeline import RAGPipeline, google_search

from Prompts import GRADING_PROMPT,HALLUCINATION_PROMPT,RELEVANT_GENERATION_PROMPT,RAG_PROMPT



class SelfRAG(RAGPipeline): # self Reflective Rag
    def __init__(self, model_name, context,count=5,retry = 3):
        super().__init__(model_name, context)
        
        self.grade_prompt = GRADING_PROMPT # ISREL : decides whether each context is relevant to the question
        self.find_hallocination = HALLUCINATION_PROMPT #  ISSUP : decide llm generation is releavant with the given context (avoid hallocination)
        self.grade_generation = RELEVANT_GENERATION_PROMPT# ISUSE : decide if generation from each chunk is useful for x
        self.count = count
        self.rag_prompt = RAG_PROMPT
        self.retry = 3
        
    def invoke(self,question):
        updated_context = self.grade_documents(question)
        generations = self.generate_remove_hallocinations(question,updated_context)
        final_generation = self.filter_answers(generations,question)
        print(question,': \n',final_generation)

        return final_generation
    def grade_documents(self,question):
        top_docs = []
        for i,docs in enumerate(self._context) :
            prompt_config = {
                "context":docs,
                "question":question
                }
            prompt = self.grade_prompt.invoke(prompt_config)
            decision = super().invoke(prompt)
            
            if(decision=="yes"):
                top_docs.append(docs)
        print("matched "+str(len(top_docs))+" documents")
        
        if(len(top_docs)>=self.count):
            return top_docs[:self.count]
        else:
            print("searching web")
            top_docs.append(google_search(question))
            return top_docs
    
    def generate_remove_hallocinations(self,question,contexts):
        print("Generating documents of context length : "+str(len(contexts)))
        top_generations = []
        for con in contexts:
            retries = self.retry
            
            prompt_config = {
                "context":con,
                "question":question
                }
            generation_prompt = self.rag_prompt.invoke(prompt_config)
            
            
            while(retries>0):
                # generate
                
                generated_answer = super().invoke(generation_prompt)
            
                #check hallocination       
                prompt_config = {
                     "context":con,
                     "generation":generated_answer
                     }
                prompt = self.find_hallocination.invoke(prompt_config)
            
                result = super().invoke(prompt)
                if(result=='yes'):
                    top_generations.append(generated_answer)
                    break
                # if its hallocinating retry generation
                retries-=1
                
        print("Generated answers after removing hallocinations : "+str(len(top_generations)))

        return top_generations
    
    def filter_answers(self,generation,question):
        filtered_Generation = []
        for asnwer in generation:
            prompt_config = {
                 "question":question,
                 "generation":asnwer
                 }
            prompt = self.grade_generation.invoke(prompt_config)
            result = super().invoke(prompt)
            
            if(result=='yes'):
                filtered_Generation.append(asnwer)
        print("Filtering unrelavant answers  : "+str(len(filtered_Generation)))

        return filtered_Generation            
    
    
            
            
            
            
             
                
        
query_prompt = "why we use Transformers?"
retrieved_docs = RAGPipeline.retrieve_docs("./vectorestore/",query_prompt,20)

 
rag = SelfRAG("llama3-8b-8192",retrieved_docs)

rag.invoke("what is transformer")
rag.invoke("what is diffusion models")





  
    
        
