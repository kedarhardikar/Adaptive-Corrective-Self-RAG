from dotenv import load_dotenv
load_dotenv()
import os
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs =[WebBaseLoader(url).load() for url in urls]
# print(docs)
doc_list = [item for sublist in docs for item in sublist]
# print(doc_list)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 600,
    chunk_overlap = 60
)

docs_split = text_splitter.split_documents(doc_list)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 1: Create vectorstore and persist it
vectorstore = Chroma.from_documents(
    documents=docs_split,
    collection_name="rag-chroma",
    embedding=embedding_function,   # 💡 Correct arg name is `embedding` here
    persist_directory="./.chroma"
)

# Step 2: Load persisted vectorstore to get retriever
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embedding_function  # only here it's `embedding_function`
).as_retriever()


# Your question
# question = "Give an agent system planning overview?"

# # Retrieve top-k relevant documents
# retrieved_docs = retriever.invoke(question)

# # Print the content of retrieved documents
# for i, doc in enumerate(retrieved_docs):
#     print(f"\n--- Chunk {i+1} ---")
#     print(doc.page_content)


# question = "Give an agent system planning overview?"

# # Embed the question manually
# query_vector = embedding_function.embed_query(question)

# # Use the Chroma client directly (bypassing retriever wrapper)
# results = vectorstore.similarity_search_by_vector(query_vector, k=4)

# for i, doc in enumerate(results):
#     print(f"\n--- Chunk {i+1} ---\n{doc.page_content}")
