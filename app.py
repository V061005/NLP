import streamlit as st
import stanza
import pandas as pd
import tempfile
import os
from pathlib import Path
import time

# Try importing speech recognition
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# Language maps for models and Google SpeechRecognition
LANG_MAP = {
    "English": "en",
    "Hindi": "hi"
}

GOOGLE_SPEECH_LANG = {
    "English": "en-IN",
    "Hindi": "hi-IN"
}

# Hindi translations for POS tags
POS_TAGS_HINDI = {
    "NOUN": "संज्ञा",
    "PROPN": "विशेष नाम",
    "VERB": "क्रिया",
    "ADJ": "विशेषण",
    "ADV": "क्रिया विशेषण",
    "PRON": "सर्वनाम",
    "ADP": "संबंधबोधक",
    "DET": "निर्देशक",
    "AUX": "सहायक क्रिया",
    "CCONJ": "समुच्चयबोधक",
    "SCONJ": "उपवाक्यबोधक",
    "PART": "अविकारी शब्द",
    "INTJ": "विस्मयादिबोधक",
    "NUM": "संख्या",
    "PUNCT": "विराम चिह्न"
}

# Set custom Stanza resource directory
os.environ["STANZA_RESOURCES_DIR"] = os.path.expanduser("~\\stanza_resources")

@st.cache_resource
def init_stanza(lang_code: str):
    """Initialize Stanza pipeline for POS tagging."""
    models_dir = os.path.join(os.getcwd(), "stanza_models")
    os.environ["STANZA_RESOURCES_DIR"] = models_dir
    try:
        stanza.download(lang_code, verbose=False)
    except Exception:
        pass
    return stanza.Pipeline(lang=lang_code, processors="tokenize,pos", use_gpu=False, verbose=False)

def transcribe_from_mic(language):
    """Record and transcribe audio from microphone."""
    if not SR_AVAILABLE:
        st.error("SpeechRecognition not installed or PyAudio missing.")
        return ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak now! Recording for a few seconds...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout=5, phrase_time_limit=8)
    try:
        st.info("🧠 Transcribing your voice...")
        text = r.recognize_google(audio, language=GOOGLE_SPEECH_LANG.get(language, "en-IN"))
        st.success("✅ Transcription complete!")
    except sr.UnknownValueError:
        text = ""
        st.warning("Could not understand your voice clearly. Try again.")
    except Exception as e:
        st.error(f"Speech recognition failed: {e}")
        text = ""
    return text

def stanza_pos(nlp, text):
    doc = nlp(text)
    rows = []
    for sent in doc.sentences:
        for word in sent.words:
            rows.append((word.text, word.upos, word.xpos if hasattr(word, "xpos") else "_"))
    return rows

def render_tagged_html(rows, show_hindi_tags=False, language="English"):
    color_map = {
        "NOUN": "#ffd966",
        "PROPN": "#f4cccc",
        "VERB": "#c9daf8",
        "ADJ": "#d9ead3",
        "ADV": "#fce5cd",
        "PRON": "#ead1dc",
        "ADP": "#cfe2f3",
        "NUM": "#fff2cc",
        "PUNCT": "#d9d2e9",
        "DET": "#e6b8af",
        "AUX": "#b6d7a8",
        "PART": "#d0e0e3",
    }
    html_parts = []
    for token, upos, xpos in rows:
        color = color_map.get(upos, "#eeeeee")
        label = POS_TAGS_HINDI.get(upos, upos) if show_hindi_tags and language == "Hindi" else upos
        html_parts.append(
            f"<span style='background:{color}; padding:3px; margin:2px; border-radius:4px; display:inline-block;'>"
            f"<strong>{token}</strong><br/><small>{label}</small></span>"
        )
    return " ".join(html_parts)

# --- Streamlit UI ---
st.set_page_config(page_title="Multilingual POS Tagger — Streamlit", layout="wide")
st.title("🎙️ NLP with GenAI — Multilingual POS Tagger")
st.markdown("""
Perform **Part-of-Speech tagging** using Stanza.  
Supports **English** and **Hindi** text input and **microphone recording**.  
""")

with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Choose Language", list(LANG_MAP.keys()))
    input_mode = st.radio("Choose Input Mode", ["Text", "Microphone"])
    show_table = st.checkbox("Show POS Table", True)
    show_html = st.checkbox("Show Colored Tags", True)
    show_hindi_tags = st.checkbox("Show POS tags in Hindi (when using Hindi)", value=False)

st.header("Input Section")
input_text = ""

if input_mode == "Text":
    input_text = st.text_area("Type or paste text below:", height=150)
else:
    if st.button("🎤 Record from Microphone"):
        input_text = transcribe_from_mic(language)
    if input_text:
        st.text_area("Transcribed Text:", value=input_text, height=150)

if st.button("Run POS Tagger") and input_text.strip():
    with st.spinner(f"Loading Stanza model for {language}..."):
        nlp = init_stanza(LANG_MAP[language])
    with st.spinner("Tagging text..."):
        rows = stanza_pos(nlp, input_text)

    if not rows:
        st.warning("No tokens found. Try again.")
    else:
        df = pd.DataFrame(rows, columns=["Token", "UPOS", "XPOS"])

        # Hindi label translation
        if show_hindi_tags and language == "Hindi":
            df["UPOS (Hindi)"] = df["UPOS"].map(lambda t: POS_TAGS_HINDI.get(t, t))

        st.subheader("Results")
        if show_table:
            st.dataframe(df)

        if show_html:
            st.markdown(
                render_tagged_html(rows, show_hindi_tags, language),
                unsafe_allow_html=True,
            )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="pos_tags.csv", mime="text/csv")

st.markdown("---")
st.caption("Built with ❤️ by Vaishnavi,Saniya,Madhumati and Revati using Streamlit, Stanza, and SpeechRecognition")
