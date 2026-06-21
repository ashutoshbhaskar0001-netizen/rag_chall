import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢")
st.title("🏢 Zyro Dynamics HR Help Desk")

# Cache data initialization so it doesn't rebuild on every UI click
@st.cache_resource
def init_rag_system():
    corpus_path = "/kaggle/input/zyro-dynamics-hr-corpus/"
    if not os.path.exists(corpus_path):
        # Fallback for local testing/Streamlit Cloud deployment
        corpus_path = "./zyro-dynamics-hr-corpus/"
        
    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

try:
    retriever = init_rag_system()
    # Read API Key from environment
    #  CORRECT (Active model)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
except Exception as e:
    st.error(f"Initialization Error: Ensure your API keys and corpus directory are accessible. Details: {e}")
    st.stop()

# Prompts
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR Help Desk Assistant for Zyro Dynamics. 
Answer the employee's question strictly based on the provided context. 
If you do not know, state clearly that you do not know.

Context:
{context}

Question: {question}
Answer:""")

OOS_PROMPT = ChatPromptTemplate.from_template("""
Analyze the question. If it is about company policies, leave, compensation, or HR topics, reply 'IN_SCOPE'. Otherwise reply 'OUT_OF_SCOPE'.
Question: {question}
Classification:""")

REFUSAL_MESSAGE = "I am sorry, but I can only assist with Zyro Dynamics HR policy and internal workplace inquiries."

# Chat Session State Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a question about Zyro Dynamics policies..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Process
    guard_chain = OOS_PROMPT | llm | StrOutputParser()
    classification = guard_chain.invoke({"question": user_query}).strip().upper()
    
    with st.chat_message("assistant"):
        if "OUT_OF_SCOPE" in classification:
            response_text = REFUSAL_MESSAGE
            st.markdown(response_text)
        else:
            retrieved_docs = retriever.invoke(user_query)
            context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            chain = RAG_PROMPT | llm | StrOutputParser()
            response_text = chain.invoke({"context": context_str, "question": user_query})
            
            st.markdown(response_text)
            
            # Show sources section elegantly
            with st.expander("📚 View Cited Sources"):
                for idx, doc in enumerate(retrieved_docs, 1):
                    source_name = os.path.basename(doc.metadata.get('source', 'Unknown Policy Document'))
                    page = doc.metadata.get('page', 0) + 1
                    st.markdown(f"**Source {idx}:** {source_name} (Page {page})")
                    st.caption(f"'{doc.page_content[:150]}...'")
                    
    st.session_state.messages.append({"role": "assistant", "content": response_text})