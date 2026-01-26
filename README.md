# 🛡️ Defence Agentic RAG Chatbot

---

🚀 **Live Demo:** [Click here to view my app](https://defence-agentic-rag.streamlit.app)

---

## Overview
This project is an **Agentic Retrieval-Augmented Generation (RAG) chatbot** designed specifically for **defence, private data**.  
It provides **grounded, document-based answers** related to:

- Air & Missile Defence Systems  
- Unmanned Aerial Systems (Drones)  
- B-2 Spirit Strategic Bomber Systems  

Unlike basic RAG systems, this solution uses **multiple domain-specific tools**, **intelligent query routing**, and **persistent memory**, making it suitable for **real-world defence AI applications**.

---

## Problem Statement
General-purpose AI models often:
- Hallucinate defence facts
- Mix unrelated domains
- Lack document grounding

In defence systems, **accuracy, grounding, and controlled responses** are critical.

---

## Objectives
- Build a **defence-restricted AI assistant**
- Prevent hallucinations using **document grounding**
- Route queries intelligently using an **agentic architecture**
- Support **multi-conversation memory**



---

##  System Architecture
The system follows an **Agentic RAG architecture** using LangGraph:

User (Streamlit UI)

  ↓

LangGraph Agent (State Controller)

  ↓

LLM + Tool Routing

  ↓

Domain-Specific RAG Tools

  ↓

Chroma Vector Databases

  ↓

Grounded Defence Answer

### Key Components
- **Frontend**: Streamlit Chat UI
- **Agent Controller**: LangGraph StateGraph
- **LLM**: LLaMA 3.3 / Gemini 
- **Vector Store**: ChromaDB
- **Memory**: SQLite 
- **Embedding Model**: Google Generative AI Embeddings

---

<img width="1536" height="1024" alt="ChatGPT Image Jan 26, 2026, 04_03_26 PM" src="https://github.com/user-attachments/assets/e053d452-1c8e-4e00-bb77-86627babef43" />


---

##  Defence Knowledge Sources
Defence Documents pdfs:

 | Document |
|---------------|
| Air Missile Defence |
| Drone Systems |
| B-2 Spirit Strategic Bombers |


Each document is:
- Loaded as PDF
- Cleaned and normalized
- Split into overlapping chunks
- Embedded and stored separately

---

## Retrieval Strategy
- **Semantic Search** using vector embeddings
- **MMR (Maximal Marginal Relevance)** for diverse and relevant results
- Each has its **own retriever & vector DB**

---

## Agentic Intelligence (LangGraph)
The chatbot uses a **state-based agent** that can:

- Decide whether to:
  - Answer directly
  - Invoke a defence-specific RAG tool
- Enforce **strict topic control**
- Reject non-defence questions
- Summarize retrieved context accurately

---

##  Memory & Conversation Handling
- Persistent memory using **SQLite**
- Thread-wise chat history
- Supports:
  - Multiple chat sessions
  - Reloading old conversations
  - Long-term contextual continuity
 
 ---

## Frontend Features (Streamlit)
- Chat-based interface
- Real-time streaming responses
- Tool execution status indicator
- Sidebar chat history
- New chat & thread switching

---

## Tech Stack
- **Python**
- **LangChain**
- **LangGraph**
- **ChromaDB**
- **Streamlit**
- **SQLite**
- **LLaMA 3 / Gemini**
- **Google Generative AI Embeddings**

