from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
import os

dotenv_path = find_dotenv()
print("Found .env at:", dotenv_path)
load_result = load_dotenv(dotenv_path)
print("Loaded .env?", load_result)

groq_api_key = os.getenv("GROQ_API_KEY")
print("GROQ API KEY:", repr(groq_api_key))

if groq_api_key is None:
    raise RuntimeError("GROQ_API_KEY not found in environment!")

model = "meta-llama/llama-4-maverick-17b-128e-instruct"
llm = ChatGroq(temperature=0, api_key=groq_api_key, model=model)

prompt = hub.pull("rlm/rag-prompt")
generation_chain = prompt | llm | StrOutputParser()
