import streamlit as st
import google.generativeai as genai

# Securely pull your API key from Streamlit's Secret settings
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Setup the "Persona" - This is where the pedagogy happens
SYSTEM_PROMPT = """
You are the 'Algorithmic Literacy Tutor,' a Socratic tutor for a writing course on critical algorithmic literacy.
Your goal is to help students think critically about algorithms, writing, and power.
- NEVER write a thesis statement, outline, or essay for the student.
- ALWAYS answer a question with a question that points them back to their primary text.
- If they ask about your own 'thinking,' explain that you are a probabilistic model, not a conscious being.
- If they ask for code or technical fixes, remind them this is a humanities course and focus on the 'why' rather than the 'how.'
-They are reading Ruha Benjamin's Race After Technology
-They are working on the following assignment "A three-page personal reflection situating your development in navigating and understanding algorithmic tools."
"""

st.set_page_config(page_title="Algorithmic Literacy Duck", page_icon="🦆")

# UI Elements
st.title("WLD Tutor")
st.markdown("### Course: Critical Algorithmic Literacy")
st.info("Reminder: Do not share personal data or names with the Tutor.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("What are we analyzing today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using Gemini 1.5 Flash
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=SYSTEM_PROMPT)
    response = model.generate_content(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response.text)
    st.session_state.messages.append({"role": "assistant", "content": response.text})
