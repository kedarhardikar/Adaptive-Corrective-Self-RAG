from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # This will find and load the .env from project root

import os
groq_api_key = os.getenv("GROQ_API_KEY")

model = "meta-llama/llama-4-maverick-17b-128e-instruct"

llm = ChatGroq(model=model, api_key=groq_api_key)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field("Are the documents relevant to the question: Yes or No")


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate(
    [
        ("system",system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader