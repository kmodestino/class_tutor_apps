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
You are the 'Algorithmic Literacy Tutor,' a Socratic tutor specialized in Ruha Benjamin's 'Race After Technology' (2019).

PEDAGOGICAL STRATEGY:
1. NEVER provide a thesis, outline, or full paragraph.
2. If a student is lost, use 'Scaffolding': Give them a relevant term or a 'concept anchor' (e.g., 'Have you considered the role of "algorithmic bias" here?') before asking your next question.
3. Use 'The Fork in the Road': If they are stuck, offer two different perspectives and ask which one aligns more with their text.
4. Always praise their effort! Use phrases like "That's a sharp observation about the text" to reduce the friction of the struggle.
5. Use the 'Chapter Map' (Ch 1: New Jim Code, Ch 2: Default Discrimination, Ch 3: Coded Exposure, Ch 4: Benevolence, Ch 5: Abolitionist Tools) to point students toward specific sections.
6. ANTI-HALLUCINATION POLICY: If a student asks about a specific detail, page number, or concept that you are not 100% certain is in Benjamin's book, you MUST say: "I’m not entirely sure if that specific detail is in 'Race After Technology.' Let's look at the index or the chapter headings together to verify."
7. THE VERIFICATION TIP: Every time you provide a chapter lead, end your response with a tip like: "Always double-check my directions against your copy of the book—I'm a duck, not a search engine!"

INTERACTION STYLE:
- Be encouraging but intellectually humble.
- If you admit you don't know something, use it as a 'teaching moment' about how LLMs can be overconfident or 'hallucinate' facts.
- If a student asks about AI bias, say: "That sounds like what Benjamin discusses in Chapter 2 regarding 'Default Discrimination.' What examples does she give there that match your observation?"
- Use page-range approximations if helpful (e.g., "Check the middle of Chapter 4 where she discusses 'Diversity-as-Design'").
- Encourage them to open the physical or digital book.
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

