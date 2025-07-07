# 🔁 Adaptive-Corrective-Self-RAG (LangGraph-Powered)

A modular, multi-stage Retrieval-Augmented Generation (RAG) pipeline built using **LangGraph**.  
This project introduces **adaptive routing**, **hallucination grading**, and **automated self-correction**, enabling robust and reliable question answering over vector-based and web data sources.

---

## 🚀 Features

- ✅ **LangGraph Workflow** with conditional and looping logic  
- 📄 **RAG Node** powered by vector search and LLM generation  
- 🌐 **Web Search Node** as fallback when internal docs are insufficient  
- 🧠 **LLM-based Graders** for:  
  - Relevance of retrieved chunks  
  - Hallucination detection  
  - Answer usefulness  
- 🔄 **Retry Loops** for hallucinated generations  
- 🧪 **Unit Tested** grading logic using `pytest`  

---

## 🧠 Architecture

```mermaid
graph TD;
  Start --> Router
  Router -->|vectorstore| Retrieve
  Router -->|websearch| WebSearch
  Retrieve --> GradeDocs
  GradeDocs -->|relevant| Generate
  GradeDocs -->|not relevant| WebSearch
  WebSearch --> Generate
  Generate --> GradeAnswer
  GradeAnswer -->|useful| End
  GradeAnswer -->|not useful| WebSearch
  GradeAnswer -->|hallucinated| Generate
