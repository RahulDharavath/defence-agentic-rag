import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash-lite')
# llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
# llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0)
# llm = ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct', temperature=0)


# -----------------------------
# VECTOR DB FOLDER
# -----------------------------
VECTOR_DB = "VECTOR_DB"
os.makedirs(VECTOR_DB, exist_ok=True)

# ---------------------------
# 1. PDF Paths
# ---------------------------

AirMissile_PDF = "Air_Missile_Defense_Systems_Engineering.pdf"
Drone_PDF = "Unmanned_Aircraft_Systems.pdf"
B2_Spirit_PDF = "B-2_Spirit_Systems_Engineering_Case_Study.pdf" 

# ----------------------------------
# 2. Text Splitter ( Into Chunks )
# ----------------------------------

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

# ----------------------------------
# 3. Create Embedding Model
# ----------------------------------
embedding = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

# ------------------------------
# Create DB / Load VectorStore
# ------------------------------
def create_db(db_path: str, pdf_path: str):
    """Load existing or create new Chroma DB with persistence."""

    if os.path.exists(db_path) and os.listdir(db_path):
        print(f"\n📌 Loading existing VectorDB: {db_path}")
        return Chroma(persist_directory = db_path, embedding_function = embedding)
    else:
        print(f"\n📌 Creating new VectorDB: {db_path}")
        
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()

        for i, doc in enumerate(docs):
            doc.page_content = doc.page_content.replace("-\n", "").replace("\n", " ")
            doc.metadata['source'] = os.path.basename(pdf_path)
            doc.metadata['page'] = i + 1
            
        chunks = splitter.split_documents(docs)

        vectordb = Chroma.from_documents(documents= chunks, embedding = embedding, persist_directory= db_path)

        
        return vectordb


# --------------------------------------------
# 4. Create Vector Store -> Chroma database
# --------------------------------------------
air_missile_vs = create_db(
    db_path = os.path.join(VECTOR_DB, "Air_Missile"),
    pdf_path = AirMissile_PDF
)


drone_vs = create_db(
    db_path = os.path.join(VECTOR_DB, "Drone"),
    pdf_path = Drone_PDF
)

b2_spirit_vs = create_db(
    db_path = os.path.join(VECTOR_DB, "B-2-Spirit"),
    pdf_path = B2_Spirit_PDF
)


# ------------------------------------------------------------------------------------
# 5. Create Retriever –> converts query to embeddings and performs semantic search
# ------------------------------------------------------------------------------------

air_missile_retriever = air_missile_vs.as_retriever(search_type= 'mmr', search_kwargs= {'k':5})

drone_retriever = drone_vs.as_retriever(search_type='mmr', search_kwargs= {'k':5})

b2_spirit_retriever = b2_spirit_vs.as_retriever(search_type='mmr', search_kwargs= {'k':5})


# ----------------------
# 6. Create Tools
# ----------------------
#   1. Air Missile RAG Tool
@tool
def air_missile_rag_tool(query: str):
    """
    Retrieve relevant information from the Air Missile Defense Systems Engineering document.
    Use this tool when the user asks factual/conceptual questions related to Air Missile Systems.
    """
    print("\n🚀 Calling AIR MISSILE SYSTEM TOOL...")
    
    result = air_missile_retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {'context' : context, 'metadata' : metadata}


## 2.Drone RAG Tool
@tool
def drone_rag_tool(query: str):

    """
    Retrieve relevant information from the Unmanned Aircraft Systems document.
    Use this tool when the user asks factual/conceptual questions related to Drones or UAS.
    """
    print("\n🛸 Calling DRONE TOOL...")
    
    result = drone_retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {'context' : context, 'metadata' : metadata}


## 3.B2 Spirit RAG Tool
@tool
def b2_spirit_rag_tool(query : str):

    """
    Retrieve relevant information from the B-2 Spirit Systems Engineering Case Study document.
    Use this tool when the user asks factual/conceptual questions related to the B-2 Spirit bomber.
    """
   
    print("\n🛩 Calling B2-Spirit TOOL...")
    
    result = b2_spirit_retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {'context' : context, 'metadata' : metadata}

#---------------------
# 7.llm_with_tools
#---------------------
tools = [air_missile_rag_tool, drone_rag_tool, b2_spirit_rag_tool]

llm_with_tools = llm.bind_tools(tools)

#-------------------------------
# 8. Create State & Agent Node
#-------------------------------
# ---------------- Agent State -------------------
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

# ---------------- LLM Agent Node ----------------

def llm_node(state : ChatState):

    SYSTEM_PROMPT = SystemMessage(content="""
    You are a DEFENSE DOMAIN EXPERT.
    
    1. ROUTING: If the user asks a factual/technical question about missiles, drones, or the B2-Spirit, use the provided  tools.
    2. DIRECT ANSWERING: For greetings or general defence talk (weapons, radar) not found in documents, answer from your general knowledge.
    3. REJECTION: For non-defense topics (sports, movies, celebrities, anything not related to defence topics), 
    just say: 'This assistant is limited to defence-related topics based on the provided documents.'
    4. GROUNDING: If you used a tool, summarize the answer accurately from the toolmessages.
    5. STRICT RULES: If tools were used but info is missing, say 'This information is not available in the documents.' 
                                  
   
""")

    messages = [SYSTEM_PROMPT] + state['messages']

    response = llm_with_tools.invoke(messages)

    return {'messages' : [response]}

                                                      
#---------------------
# 9. Tool Node --> it is a built-in pre-built node designed to automate the execution of tools.
#---------------------
tool_node = ToolNode(tools)

# ----------------------
# 10. Memory
# ----------------------
conn = sqlite3.connect(database="Rag_Memory.db", check_same_thread= False)
memory = SqliteSaver(conn)

# ---------------------------
# 11. Build Graph
# ---------------------------

graph = StateGraph(ChatState)

# Nodes
graph.add_node('llm', llm_node)
graph.add_node('tools', tool_node)


# Edges
graph.add_edge(START, 'llm')
graph.add_conditional_edges('llm', tools_condition)  # Chat node decides: Use tool? Or answer directly?
graph.add_edge('tools', 'llm') 


chatbot = graph.compile(checkpointer= memory)

# -----------------------
# 12. Threads id
# -----------------------
def retrieve_all_threads():   # counting unqiue threads
    all_threads = set()
    for checkpoint in memory.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return(list(all_threads))

