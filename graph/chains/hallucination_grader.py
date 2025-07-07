from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # This will find and load the .env from project root

import os
groq_api_key = os.getenv("GROQ_API_KEY")

model = "meta-llama/llama-4-maverick-17b-128e-instruct"
# print("GROQ API Key:", repr(groq_api_key))

model = "meta-llama/llama-4-maverick-17b-128e-instruct"

llm = ChatGroq(temperature=0, api_key= groq_api_key,model = model)

class GradeHallucination(BaseModel):
    """BINARY SCORE FOR HALLUCINATIONS IN THE ANSWER"""

    binary_score: bool = Field(
        description= "Answer is grounded in the facts, 'yes" or 'no '
        )
    

structured_llm_grader = llm.with_structured_output(GradeHallucination)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader