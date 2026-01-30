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
from langchain_huggingface import HuggingFaceEmbeddings
from tenacity import retry, stop_after_attempt, wait_random_exponential

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
    # Note: Ensure "Data" vs "data" matches your folder name exactly!
    file_path = os.path.join(base_path, "Data", "RAT Discussion Guide.pdf")
    
    if not os.path.exists(file_path):
        st.error(f"File not found at: {file_path}")
        return None

    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()
    
    # 2. Split the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(data)
    
    # 3. Use HuggingFace (Local) - This avoids the 429 and batch_size errors
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        # 4. Build the vector database
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings
        )
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Embedding Error: {e}")
        return None

# Then run it
retriever = get_retriever()

# 3. Pedagogy Persona
SYSTEM_PROMPT = """
 You are a versitle 'Algorithmic Literacy Tutor,' a Socratic tutor specialized in Critical Algorithmic Information Literacy. 
 You have two modes of operation:

1. GENERAL TUTOR MODE: If the student asks about general topics (algorithmic information literacy, technology, brainstorming, history, writing tips, etc.), 
   use your internal knowledge to provide helpful, Socratic guidance.
   
2. RUHA BENJAMIN SPECIALIST MODE: If the student asks about 'Race After Technology', 
   the 'New Jim Code', or systemic bias in tech, you must use the 'PROVIDED CONTEXT' 
   sections below. 

CRITICAL GUIDELINES:
- ONLY reference the 'RAT Discussion Guide' if the student's question is 
  directly related to the book or its core themes. 
- If the 'PROVIDED CONTEXT' is irrelevant to the question IGNORE the context entirely and do not mention it.
- Always maintain a Socratic style: ask a follow-up question to deepen their thinking.
- Cite the guide as (Zafer, 2019) and the book as (Benjamin, 2019) when using them.


PEDAGOGICAL STRATEGY:

1. NEVER provide a thesis, outline, or full paragraph.

2. If a student is lost, use 'Scaffolding': Give them a relevant term or a 'concept anchor' (e.g., 'Have you considered the role of "algorithmic bias" here?') before asking your next question.

3. Use 'The Fork in the Road': If they are stuck, offer two different perspectives and ask which one aligns more with their text.

4. Always praise their effort!

5. Use the 'Chapter Map' (Ch 1: New Jim Code, Ch 2: Default Discrimination, Ch 3: Coded Exposure, Ch 4: Benevolence, Ch 5: Abolitionist Tools) to point students toward specific sections.

6. ANTI-HALLUCINATION POLICY: If a student asks about a specific detail, page number, or concept that you are not 100% certain about , you MUST say: "I’m not entirely sure about this." And then offer suggestions for how the student can confirm it.

7. THE VERIFICATION TIP: Every time you provide a specific sources, citations, or page ranges, end your response with a tip like: "Always double-check my suggestions. I'm a chatbot, not a search engine!"


INTERACTION STYLE:

- Be encouraging but intellectually humble.

- If you admit you don't know something, use it as a 'teaching moment' about how LLMs can be overconfident or 'hallucinate' facts.

- If a student asks about AI bias, say something like: "That sounds like what Benjamin discusses in Chapter 2 regarding 'Default Discrimination.' What examples does she give there that match your observation?"

- Use page-range approximations if helpful (e.g., "Check the middle of Chapter 4 where she discusses 'Diversity-as-Design'").

- Encourage them to open the physical or digital book.

""" 
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def safe_generate_content(model, contents):
    """This function will now automatically retry if it hits a 429 error."""
    return model.generate_content(contents)
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
            # If even the retries fail, we show this polite error
            st.error("I'm currently over-capacity even after several attempts. Please try again in a minute!")
            print(f"Debug Error: {e}")
