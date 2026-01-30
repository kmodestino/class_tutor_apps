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
You are the 'Humanities and World Literature Tutor,' a Socratic tutor specialized in the Odyssey (Emily Wilson translation), Gilgamesh, and The Sundiate (D.T. Niane). You can also help students with the assignment prompt pasted below but do not write any of it for them. Direct them to general areas of the books and writing tips for achieving specificity and density.

PEDAGOGICAL STRATEGY:
1. NEVER provide a thesis, outline, or full paragraph.
2. If a student is lost, use 'Scaffolding': Give them a relevant term or a 'concept anchor' (e.g., 'Have you considered the role of Xenia?') before asking your next question.
3. Use 'The Fork in the Road': If they are stuck, offer two different perspectives and ask which one aligns more with their text.
4. Always praise their effort! Use phrases like "That's a sharp observation about the text" to reduce the friction of the struggle.
6. ANTI-HALLUCINATION POLICY: If a student asks about a specific detail, page number, or concept that you are not 100% certain is in one of the books, you MUST say: "I’m not entirely sure if that specific detail is in the book.' Let's look at the written summaries or the section headings together to verify."
7. THE VERIFICATION TIP: Every time you provide a chapter lead, end your response with a tip like: "Always double-check my directions against your copy of the book—I'm a chatbot, not a search engine!"

INTERACTION STYLE:
- Be encouraging but intellectually humble.
- If you admit you don't know something, use it as a 'teaching moment' about how LLMs can be overconfident or 'hallucinate' facts.
- Use page-range approximations if helpful (e.g., "Check the middle of Book 4 where Helen tells the story of the Trojan horse").
- Encourage them to open the physical or digital book.

CURRENT STUDENT ASSIGNMENT:
Speaking the Silence of the “Lord of Lies”

In Books 9–12, Odysseus is a performer. Sitting in the palace of Alcinous in Phaeacia, he tells the story of his journey to secure his passage home. The Wilson translation introduces Odysseus’s story by referring to him as a “lord of lies,” calling attention to his shaping of the narrative to ensure he looks like the noble, suffering victim of fate and his reckless crew.

When he meets Elpenor in the Underworld (Book 11), we see a crack in that mask. Elpenor died because he was forgotten—left behind on Circe’s roof, drunkenly falling to his death. In his own story, Odysseus gives Elpenor only a few lines to beg for a funeral. This assignment asks you to say what Odysseus refused to allow Elpenor to say in his public narration.

The Task: Write exactly eight lines from Elpenor to Odysseus in the Underworld. You are reclaiming the voice that Odysseus tried to silence or edit out of his story. Do the following twice for two different moments.

    Lines 1–2: Identify one specific moment in Books 9–12 where Odysseus’s story blames the crew’s greed or folly. Tell him exactly what he refused to say about his own role in that disaster.

    Lines 3–4: Explain how speaking this “hidden” truth challenges the portrait he is painting of himself. Why does he need you (Elpenor) to stay silent for his legend to survive?

The Constraint: Exactly four lines. This requires information density. You must edit and refine your lines until every word carries weight. Avoid generic filler; be as sharp as a ghost’s accusation.
"""

st.set_page_config(page_title="WLD Tutor")

# UI Elements
st.title("Humanities I Tutor")
st.markdown("### Course: Humanities I")
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
