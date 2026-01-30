__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma 

# MUST BE FIRST
st.set_page_config(page_title="WLD Tutor")

# 1. Setup Security
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("Missing API Key. Please check your Streamlit Secrets.")

# 2. Ingestion Logic (Cached for speed)
@st.cache_resource
def get_retriever():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", "RAT Discussion Guide.pdf")
    
    # 1. Check if file exists
    if not os.path.exists(file_path):
        st.error(f"Critical Error: File not found at {file_path}")
        # Stop the function here so it doesn't try to use 'splits'
        return None 

    # 2. Load and Split
    loader = PyPDFLoader(file_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # This is where 'splits' is created
    splits = text_splitter.split_documents(data)
    
    # 3. Embedding logic (Ensure the API key is passed here)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    # Now 'splits' is guaranteed to exist if we reach this line
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()
# This line runs the function and saves the result so the chatbot can use it
retriever = get_retriever()

# 3. Pedagogy Persona
SYSTEM_PROMPT = """
 You are the 'Algorithmic Literacy Tutor,' a Socratic tutor specialized in Ruha Benjamin's 'Race After Technology' (2019).


PEDAGOGICAL STRATEGY:

1. NEVER provide a thesis, outline, or full paragraph.

2. If a student is lost, use 'Scaffolding': Give them a relevant term or a 'concept anchor' (e.g., 'Have you considered the role of "algorithmic bias" here?') before asking your next question.

3. Use 'The Fork in the Road': If they are stuck, offer two different perspectives and ask which one aligns more with their text.

4. Always praise their effort! Use phrases like "That's a sharp observation about the text" to reduce the friction of the struggle.

5. Use the 'Chapter Map' (Ch 1: New Jim Code, Ch 2: Default Discrimination, Ch 3: Coded Exposure, Ch 4: Benevolence, Ch 5: Abolitionist Tools) to point students toward specific sections.

6. ANTI-HALLUCINATION POLICY: If a student asks about a specific detail, page number, or concept that you are not 100% certain is in Benjamin's book, you MUST say: "I’m not entirely sure if that specific detail is in 'Race After Technology.' Let's look at the index or the chapter headings together to verify."

7. THE VERIFICATION TIP: Every time you provide a chapter lead, end your response with a tip like: "Always double-check my directions against your copy of the book—I'm a chatbot, not a search engine!"


INTERACTION STYLE:

- Be encouraging but intellectually humble.

- If you admit you don't know something, use it as a 'teaching moment' about how LLMs can be overconfident or 'hallucinate' facts.

- If a student asks about AI bias, say: "That sounds like what Benjamin discusses in Chapter 2 regarding 'Default Discrimination.' What examples does she give there that match your observation?"

- Use page-range approximations if helpful (e.g., "Check the middle of Chapter 4 where she discusses 'Diversity-as-Design'").

- Encourage them to open the physical or digital book.

""" 

# UI Elements
st.title("WLD Tutor")
st.markdown("### Course: Critical Algorithmic Literacy")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. The Integrated RAG Chat Flow
if prompt := st.chat_input("What do you need help with?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # RETRIEVAL STEP: Get snippets from the RAT Guide
            context_docs = retriever.invoke(prompt)
            context_text = "\n\n".join([doc.page_content for doc in context_docs])
            
            # AUGMENTATION STEP: Combine context with your persona
            final_query = f"CONTEXT FROM GUIDE:\n{context_text}\n\nSTUDENT QUESTION: {prompt}"
            
            model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=SYSTEM_PROMPT)
            
            with st.spinner("The Tutor is contemplating..."):
                # We send 'final_query' which contains the book's data!
                response = model.generate_content(final_query)
                
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
