from typing import Any, Dict
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # This will find and load the .env from project root


import os
groq_api_key = os.getenv("GROQ_API_KEY")
from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation} 