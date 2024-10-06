# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 06:02:23 2024

@author: mitran
"""

from RAGPipeline import RAGPipeline, google_search
from sklearn.cluster import KMeans
from collections import defaultdict,Counter
from Prompts import DRAFTING_PROMPT,VERIFY_DRAFTING_PROMPT
import asyncio
import random

import nest_asyncio
nest_asyncio.apply()

import regex as re
def extract_substring(input_string):
    match = re.search(r'##.*?:', input_string)
    return match.group(0).strip() if match else False

def multiPerspectiveSampleing(docs,k:int=4,no_drafts=None):
    
    
    
    assert len(docs) >= k
    
    kmeans  = KMeans(n_clusters=k,random_state=2727)
    vectors = [doc.vector for doc in docs]
    clusters = kmeans.fit_predict(vectors)
    
    print(clusters)
    docs_cluster = defaultdict(list)
    for ind in clusters:
        docs_cluster[ind].append(docs[ind].page_content)
    m = no_drafts if no_drafts else min(Counter(clusters).values()) 
    drafts = []
    for _ in range(m):
        draft = []
        for key in docs_cluster.keys():
            sample = docs_cluster[key]
            doc = random.choice(sample)
            sample.remove(doc)
            draft.append(doc)
        
        drafts.append(draft)
    return drafts    
    

class DraftRAG(RAGPipeline):
    def __init__(self, model_name, drafts):
        super().__init__(model_name, None) # small model
        
        self.draft_prompt = DRAFTING_PROMPT
        self.verify_draft_prompt = VERIFY_DRAFTING_PROMPT
        self.drafts= drafts
    def invoke(self,question):
        print("started drafting)")
        generted_rag_draft = asyncio.run(self.run_async_drafts(question,self.drafts))
        print("drafted")
        verify_rationale = asyncio.run(self.run_async_Verify_drafts(question,generted_rag_draft))
        print("verified")
        return verify_rationale
        
    async def run_async_drafts(self,question,docs):
        draft_task = []
        for draft_doc in docs:
            prompt_config = {
                "question":question,
                "draft":draft_doc
                }
            prompt =  self.draft_prompt.invoke(prompt_config)
            result = await self.invokeAsync(prompt)
            draft_task.append(result)
        return draft_task
    async def run_async_Verify_drafts(self,question,generted_rag_draft):
        draft_task = []

        for draft in generted_rag_draft:
            structured_llm_output = self.parse_llm_output(draft)
            prompt_config = {
                "question":question,
                "response":structured_llm_output["Response"],
                "rationale":structured_llm_output["Rationale"]
                }
            prompt = self.verify_draft_prompt.invoke(prompt_config)
            result = await self.invokeAsync(prompt)
            print(result,"<---")
            draft_task.append(result)
        return draft_task
            
    def parse_llm_output(self,llm_output):
        parts = [line for line in llm_output.strip().split('\n') if len(line) > 0]
        structured_llm_output = defaultdict(list)
        key = "default"
        for part in parts:
            if(part.startswith("##")):
                key = extract_substring(part)
                structured_llm_output[key.replace('## ','').replace(':','')].append(part.replace(key,''))
                key = key.replace('## ','').replace(':','')
            else:
                
                structured_llm_output[key].append(part)
        for k,v in structured_llm_output.items():
            structured_llm_output[k] = '\n'.join(v).strip()
        assert all(key in structured_llm_output.keys() for key in ['Rationale','Response'])
        return structured_llm_output
            
        
    

    
            
            
    
# retrieve many docs with the vectors
query_prompt = "what is Transformers?"
retrieved_docs = RAGPipeline.retrieve_docs("./vectorestore/",query_prompt,15,with_vectors=True)
drafts = multiPerspectiveSampleing(retrieved_docs)
draftRag = DraftRAG("llama-3.2-3b-preview",drafts)
y = draftRag.invoke(query_prompt)







