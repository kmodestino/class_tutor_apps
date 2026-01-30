import streamlit as st
import google.generativeai as genai
import time

# 1. Setup Security & Model
# Pulls the key from your Streamlit Secrets box
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error("Missing API Key. Please check your Streamlit Secrets.")
    
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

st.set_page_config(page_title="WLD Tutor")

# UI Elements
st.title("WLD Tutor")
st.markdown("### Course: Critical Algorithmic Literacy")
st.info("Reminder: Do not share personal data or names with the Tutor.")

# 3. Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Input & Rate Limit Handling
if prompt := st.chat_input("What do you need help with?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Using the latest 2026 stable model
            model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=SYSTEM_PROMPT)
            
            # Add a small 'thinking' delay to mimic human rhythm
            with st.spinner("The Tutor is contemplating..."):
                response = model.generate_content(prompt)
                
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            # Check for the common 'Rate Limit' error (429)
            if "429" in str(e) or "quota" in str(e).lower():
                st.error("⚠️ **The Tutor is overwhelmed!** Too many students are asking questions at once. Please wait 60 seconds for the 'Free Tier' quota to reset.")
            else:
                st.error(f"An unexpected error occurred: {e}")

