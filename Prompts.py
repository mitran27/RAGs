# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:53:18 2024

@author: mitran
"""

from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain import hub


GRADING_PROMPT:PromptTemplate =  PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n
    Here is the retrieved document: \n\n {context} \n\n
    Here is the user question: {question} \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\n
    provide string 'yes' or'no' only""",
    input_variables=["context", "question"],
)


HALLUCINATION_PROMPT:PromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a grader assessing whether an LLM generation is supported by a set of retrieved facts. \n 
             Give a binary score 'yes' or 'no'. 'Yes' means that the answer is supported by the set of facts.
             provide string 'yes' or'no' only"""),
        ("human", "Set of facts: \n\n {context} \n\n LLM generation: {generation}"),
    ]
)
         
RELEVANT_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a grader assessing whether an answer addresses / resolves a question \n 
             Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
             provide string 'yes' or'no' only"""),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
         
RAG_PROMPT = hub.pull("rlm/rag-prompt")


DRAFTING_PROMPT: PromptTemplate = PromptTemplate(
    template="""Response to the instruction using the Evidence. Also provide rationale for your response.
## Instruction: {question}

## Evidence: {draft}""",input_variables=["question","draft"])

VERIFY_DRAFTING_PROMPT: PromptTemplate = PromptTemplate(
    template="""## Instruction: {question}

## Response: {response} 

## Rationale: {rationale}

Is the rationale good enough to support the answer? (Yes or No)""",input_variables=["question","response","rationale"])