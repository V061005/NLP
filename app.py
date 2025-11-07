import streamlit as st
import stanza
import pandas as pd
import os
import language_tool_python

# Try importing speech recognition
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# Supported languages for Stanza POS tagging
LANG_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur"
}

# Google Speech API language codes
GOOGLE_SPEECH_LANG = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Bengali": "bn-IN",
    "Marathi": "mr-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Urdu": "ur-IN"
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

# Stanza resource directory
os.environ["STANZA_RESOURCES_DIR"] = os.path.expanduser("~\\stanza_resources")

SUPPORTED_LANGS = ["en", "hi", "bn", "mr", "ta", "te", "ur"]

@st.cache_resource
def init_stanza(lang_code: str):
    """Initialize Stanza pipeline for POS tagging."""
    if lang_code not in SUPPORTED_LANGS:
        st.warning(f"⚠ POS tagging is not available for '{lang_code}'.")
        return None
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
        st.info("🎙 Speak now! Recording for a few seconds...")
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

def check_grammar(text, language):
    if language != "English":
        return text, pd.DataFrame([{"Message": "⚠ Grammar correction is available only for English."}])
    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    error_list = [{"Issue": m.ruleId, "Message": m.message, "Incorrect Text": m.context,
                   "Suggestion": ", ".join(m.replacements) if m.replacements else "-"} for m in matches]
    return corrected, pd.DataFrame(error_list)

def stanza_pos(nlp, text):
    doc = nlp(text)
    rows = [(w.text, w.upos, w.xpos if hasattr(w, "xpos") else "_") for sent in doc.sentences for w in sent.words]
    return rows

def render_tagged_html(rows, show_hindi_tags=False, language="English"):
    color_map = {
        "NOUN": "#e3f2fd", "PROPN": "#fce4ec", "VERB": "#e8f5e8", "ADJ": "#fff3e0",
        "ADV": "#f3e5f5", "PRON": "#ede7f6", "ADP": "#e0f2f1", "NUM": "#fafafa",
        "PUNCT": "#f5f5f5", "DET": "#ffebee", "AUX": "#e8f5e8", "PART": "#f1f8e9",
    }
    html_parts = []
    for token, upos, xpos in rows:
        color = color_map.get(upos, "#f5f5f5")
        label = POS_TAGS_HINDI.get(upos, upos) if show_hindi_tags and language == "Hindi" else upos
        # Improved contrast: Black text for visibility
        html_parts.append(f"<span style='background:{color}; color: #333; padding:4px 8px; margin:2px; border-radius:6px; display:inline-block; border:1px solid #ccc; font-weight: bold; transition: all 0.3s;'>"
                          f"{token}<br/><small style='color:#555; font-weight: normal;'>{label}</small></span>")
    return " ".join(html_parts)

# --- Streamlit UI ---
st.set_page_config(page_title="NLP POS Tagger", layout="wide", page_icon="🔬")
# ================================
# 🌐 GLOBAL SETTINGS (Enhanced)
# ================================
with st.sidebar:
    st.markdown("""
        <h2 style='text-align:center; color:#00b4d8;'>⚙ Global Settings</h2>
        <hr style="margin-top: -10px; margin-bottom: 15px;">
    """, unsafe_allow_html=True)

    # Dark Mode Toggle
    dark_mode = st.toggle("🌙 Enable Dark Mode", value=False)

    # Language Selector
    st.markdown("#### 🌐 Language Settings")
    language = st.selectbox(
        "Select Language",
        list(LANG_MAP.keys()),
        help="Choose the language for text processing and analysis."
    )

    # Input Mode
    st.markdown("#### Input Preferences")
    input_mode = st.radio(
        "Input Mode",
        ["Text", "Microphone"],
        index=0,
        help="Type manually or use your voice for transcription."
    )

    # Display Options
    st.markdown("#### 🎨 Display Preferences")
    show_table = st.checkbox(
        "📊 Show POS Table",
        True,
        help="Display detailed POS tagging results in a structured table."
    )
    show_html = st.checkbox(
        "💡 Show Colored Tags",
        True,
        help="Enable visually highlighted POS elements within text."
    )
    show_hindi_tags = st.checkbox(
        "🇮🇳 Show Hindi POS Labels",
        value=True if language == "Hindi" else False,
        help="Display Hindi equivalents for POS tags (Hindi only)."
    )

    # Divider
    st.markdown("<hr>", unsafe_allow_html=True)

    # About / Info box
    with st.expander("ℹ About This App"):
        st.markdown("""
        - *PolyLingua NLP Studio*  
          Multilingual AI-powered text analysis tool.
        - Perform POS tagging, sentiment analysis, grammar check, and more.
        - Built with *Streamlit, **Stanza, and **Transformers*.
        """)

    # Footer Credits
    st.markdown("""
        <div style='text-align:center; font-size:13px; margin-top:15px; opacity:0.8;'>
        Made with ❤ by <b>Vaishnavi, Saniya, Madhumati & Revati</b>
        </div>
    """, unsafe_allow_html=True)


# 🌙 Dynamic sidebar theme styling (dark / light)
st.markdown(f"""
    <style>
        /* Textarea Styling */
        textarea {{
            color: {'#ffffff' if dark_mode else '#000000'} !important;
            background-color: {'#1e1e1e' if dark_mode else '#ffffff'} !important;
        }}

        /* Sidebar general text */
        section[data-testid="stSidebar"] {{
            color: {'#f0f0f0' if dark_mode else '#222222'} !important;
            background-color: {'#111827' if dark_mode else '#f9f9f9'} !important;
        }}

        /* Sidebar labels and headers */
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label {{
            color: {'#f9fafb' if dark_mode else '#222222'} !important;
        }}

        /* Sidebar selectboxes, checkboxes, and radio buttons */
        div[data-testid="stSidebar"] div[role="radiogroup"] label, 
        div[data-testid="stSidebar"] div[role="checkbox"] label {{
            color: {'#e5e7eb' if dark_mode else '#222222'} !important;
        }}

        /* Sidebar section dividers */
        hr {{
            border-color: {'#444' if dark_mode else '#ddd'} !important;
        }}
    </style>
""", unsafe_allow_html=True)


# Custom CSS for Full Page Theme Support
css = f"""
    <style>
        body {{ background-color: {'#121212' if dark_mode else '#ffffff'}; color: {'#ffffff' if dark_mode else '#333333'}; }}
        .main {{ background-color: {'#1e1e1e' if dark_mode else '#f9f9f9'}; }}
        .hero {{ background: {'linear-gradient(135deg, #00695c 0%, #004d40 50%, #00251a 100%)' if dark_mode else 'linear-gradient(135deg, #1976d2 0%, #0d47a1 50%, #1565c0 100%)'}; color: white; padding: 40px; border-radius: 12px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }}
        .hero h1 {{ font-size: 2.5em; margin-bottom: 10px; font-weight: 600; }}
        .hero p {{ font-size: 1.1em; margin-bottom: 15px; }}
        .feature-card {{ background: {'#424242' if dark_mode else '#ffffff'}; color: {'#ffffff' if dark_mode else '#333333'}; padding: 20px; border-radius: 10px; margin: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border: 1px solid {'#666' if dark_mode else '#ddd'}; transition: transform 0.2s; }}
        .feature-card:hover {{ transform: translateY(-5px); }}
        .tab-content {{ padding: 20px; background: {'#2c2c2c' if dark_mode else '#ffffff'}; color: {'#ffffff' if dark_mode else '#333333'}; border-radius: 10px; margin-top: 10px; }}
        .footer {{ text-align: center; margin-top: 40px; color: {'#cccccc' if dark_mode else '#777'}; font-size: 0.9em; }}
        .stButton button {{ border-radius: 8px; font-weight: 500; transition: background 0.3s; }}
        .stButton button:hover {{ background-color: #1976d2 !important; color: white !important; }}
        .stTextArea textarea, .stSelectbox select, .stRadio label {{ color: {'#ffffff' if dark_mode else '#333333'} !important; }}
        .stDataFrame, .stTable {{ background: {'#2c2c2c' if dark_mode else '#ffffff'} !important; color: {'#ffffff' if dark_mode else '#333333'} !important; }}
    </style>
"""

st.markdown(css, unsafe_allow_html=True)  # Apply the CSS

# Tab-Based Navigation
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Home", 
    "🏷 POS Tagger", 
    "📝 Grammar Check", 
    "💬 Sentiment Analysis", 
    "🧠 Entity Recognition", 
    "☁ Word Cloud & Frequency", 
    "📘 Text Summarizer"
])


# ==========================================
# Tab 1: Home (Updated)
# ==========================================
with tab1:
    st.markdown(f"""
        <div class="hero">
            <h1>🧠 PolyLingua NLP Studio</h1>
            <p>Empowering multilingual understanding through AI-driven Natural Language Processing.</p>
            <p>Analyze, visualize, correct, summarize, and speak — all in one intelligent platform.</p>
            <p><b>Created with ❤ by Vaishnavi, Saniya, Madhumati, and Revati</b></p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Explore Key Features")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h4>🏷 POS Tagger</h4>
                <p>Analyze grammatical structures across English and Indian languages with Stanza NLP.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="feature-card">
                <h4>🧠 Entity Recognition</h4>
                <p>Automatically detect and label entities like people, organizations, and locations.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="feature-card">
                <h4>☁ Word Cloud</h4>
                <p>Visualize your text’s most frequent words beautifully and intuitively.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="feature-card">
                <h4>📝 Grammar Correction</h4>
                <p>Fix English grammar errors instantly with AI-based LanguageTool integration.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="feature-card">
                <h4>💬 Sentiment Analysis</h4>
                <p>Measure emotional tone — Positive, Negative, or Neutral — with TextBlob and Plotly gauge charts.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="feature-card">
                <h4>📘 Text Summarizer</h4>
                <p>Condense long paragraphs into concise summaries using Transformer models.</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="feature-card">
                <h4>🎙 Speech Recognition</h4>
                <p>Convert voice to text in multiple Indian languages using Google Speech API.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="feature-card">
                <h4>🌐 Multilingual Support</h4>
                <p>Works seamlessly with English, Hindi, Bengali, Marathi, Tamil, Telugu, and Urdu.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Quick Start Guide")
    st.markdown("""
    1. *Select Language* from the sidebar.  
    2. *Input Text or Use Microphone* for speech-to-text.  
    3. Navigate through tabs:
        - POS Tagger — Part-of-speech breakdown  
        - Grammar Check — Correct English text  
        - Sentiment Analyzer — Detect tone  
        - Entity Recognition — Find people, places, and orgs  
        - Word Cloud — Visualize word frequency  
        - Text Summarizer — Get concise summaries  
    4. *Download Results* as CSV or text files.
    """)

# Tab 2: POS Tagger
with tab2:
    st.header("🏷 Part-of-Speech Tagger")

    # --- Input mode: text or voice ---
    if input_mode == "Text":
        input_text = st.text_area(
            "Enter or paste text:",
            height=150,
            placeholder="Type your text here...",
            key="pos_text_input"
        )
    else:
        st.write("🎤 Click to Record")
        if st.button("🎙 Start Recording", key="pos_record_button"):
            spoken = transcribe_from_mic(language)
            st.success(f"✅ You said: {spoken}")
            st.session_state["voice_text"] = spoken

        input_text = st.session_state.get("voice_text", "")
        st.text_area(
            "Transcribed Text:",
            value=input_text,
            height=150,
            disabled=True,
            key="pos_voice_input"
        )

    # --- Analyze POS button (now visible for both modes) ---
    if st.button("🔍 Analyze POS", key="pos_analyze") and input_text.strip():
        nlp = init_stanza(LANG_MAP[language])

        if nlp is not None:
            progress = st.progress(0)
            with st.spinner("Processing..."):
                for i in range(100):
                    progress.progress(i + 1)
                rows = stanza_pos(nlp, input_text)

            if not rows:
                st.error("No tokens detected. Check input.")
            else:
                df = pd.DataFrame(rows, columns=["Token", "UPOS", "XPOS"])
                if show_hindi_tags and language == "Hindi":
                    df["UPOS (Hindi)"] = df["UPOS"].map(lambda t: POS_TAGS_HINDI.get(t, t))

                col1, col2 = st.columns(2)
                with col1:
                    if show_table:
                        st.dataframe(df, use_container_width=True)
                with col2:
                    if show_html:
                        st.markdown("*Visual POS Tags:*")
                        st.markdown(render_tagged_html(rows, show_hindi_tags, language), unsafe_allow_html=True)

                # --- Statistics ---
                st.subheader("📊 Statistics")
                pos_counts = df["UPOS"].value_counts()
                total = pos_counts.sum()
                percentage = (pos_counts / total * 100).round(2)
                stats_df = pd.DataFrame({
                    "Tag": pos_counts.index,
                    "Count": pos_counts.values,
                    "%": percentage.values
                })
                st.dataframe(stats_df, use_container_width=True)

                # --- Charts ---
                import plotly.express as px
                stats_df_sorted = stats_df.sort_values("Count", ascending=False)
                chart_tab1, chart_tab2 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart"])
                with chart_tab1:
                    fig_bar = px.bar(
                        stats_df_sorted, x="Count", y="Tag", orientation="h",
                        text=stats_df_sorted["%"].apply(lambda x: f"{x:.1f}%"),
                        color="Count", color_continuous_scale="Tealgrn" if dark_mode else "Blues",
                        title="POS Tag Frequency Distribution"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                with chart_tab2:
                    fig_pie = px.pie(
                        stats_df_sorted, names="Tag", values="Count",
                        title="POS Tag Proportion (%)", hole=0.4
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                # --- Download button ---
                st.download_button(
                    "⬇ Export CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "pos_tags.csv",
                    "text/csv"
                )

# Tab 3: Grammar Check
with tab3:
    st.header("📝 Grammar Correction")
    if language != "English":
        st.warning("Grammar check is only available for English.")
    else:
        input_text = st.text_area("Enter text for grammar check:", height=150, placeholder="Paste English text here...")
        if st.button("🔍 Check Grammar", key="grammar_check") and input_text.strip():
            progress = st.progress(0)
            with st.spinner("Analyzing..."):
                for i in range(100):
                    progress.progress(i + 1)
                corrected, errors = check_grammar(input_text, language)
            
            st.subheader("Corrected Text")
            st.write(corrected)
            st.subheader("Issues Found")
            if len(errors) > 0:
                st.dataframe(errors, use_container_width=True)
            else:
                st.success("No errors detected!")

# ================================
# Tab 4: Sentiment Analysis
# ================================
with tab4:
    st.header("💬 Sentiment Analysis")

    # --- Input mode: text or voice ---
    if input_mode == "Text":
        input_text = st.text_area(
            "Enter text for sentiment analysis:",
            height=150,
            placeholder="Type or paste your text here..."
        )
    else:
        st.write("🎤 Click to Record")
        if st.button("🎙 Start Recording (Sentiment)"):
            spoken = transcribe_from_mic(language)
            st.success(f"✅ You said: {spoken}")
            st.session_state["voice_text_sent"] = spoken

        input_text = st.session_state.get("voice_text_sent", "")
        st.text_area("Transcribed Text:", value=input_text, height=150, disabled=True)

    # --- Analyze Sentiment button ---
    if st.button("🔍 Analyze Sentiment", key="sentiment_analyze") and input_text.strip():
        try:
            from textblob import TextBlob
            import plotly.graph_objects as go

            blob = TextBlob(input_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Interpret sentiment
            if polarity > 0.1:
                sentiment_label = "Positive 😀"
                color = "green"
            elif polarity < -0.1:
                sentiment_label = "Negative 😞"
                color = "red"
            else:
                sentiment_label = "Neutral 😐"
                color = "gray"

            # --- Display results ---
            st.markdown(f"### Sentiment: *{sentiment_label}*")
            st.write(f"*Polarity:* {polarity:.2f}")
            st.write(f"*Subjectivity:* {subjectivity:.2f}")

            # --- Gauge visualization ---
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=polarity,
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [-1, -0.1], 'color': "#ff9999"},
                        {'range': [-0.1, 0.1], 'color': "#d9d9d9"},
                        {'range': [0.1, 1], 'color': "#90ee90"},
                    ],
                },
                title={'text': "Sentiment Polarity (-1 to +1)"}
            ))
            st.plotly_chart(fig, use_container_width=True)

            # --- Summary tips ---
            if polarity > 0.5:
                st.success("🌞 Very positive tone — good for marketing or friendly messages.")
            elif polarity < -0.5:
                st.warning("⚠ Strong negative tone — could be expressing criticism or sadness.")
            else:
                st.info("😐 Balanced tone — mostly neutral or factual.")

            # --- Download report ---
            st.download_button(
                "⬇ Export Sentiment Report",
                f"Text: {input_text}\nPolarity: {polarity}\nSubjectivity: {subjectivity}\nSentiment: {sentiment_label}".encode("utf-8"),
                "sentiment_report.txt",
                "text/plain"
            )

        except ImportError:
            st.error("Please install TextBlob and Plotly: pip install textblob plotly")


# ================================
# Tab 5: Entity Recognition
# ================================
import spacy

with tab5:
    st.header("🧠 Named Entity Recognition (NER)")
    st.write("Automatically identify and categorize named entities like names, organizations, and locations.")

    input_text = st.text_area("Enter text for entity recognition:", height=150, placeholder="Type or paste English text here...")
    if st.button("🔍 Extract Entities"):
        try:
            nlp = spacy.load("en_core_web_trf")
            doc = nlp(input_text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            if entities:
                df_ner = pd.DataFrame(entities, columns=["Entity", "Label"])
                st.dataframe(df_ner, use_container_width=True)

                # Show highlighted text
                html = spacy.displacy.render(doc, style="ent", jupyter=False)
                st.markdown(html, unsafe_allow_html=True)

                # Download results
                st.download_button(
                    "⬇ Export Entities",
                    df_ner.to_csv(index=False).encode("utf-8"),
                    "entities.csv",
                    "text/csv"
                )
            else:
                st.info("No named entities found in the text.")
        except Exception as e:
            st.error(f"Error: {e}")


# ================================
# Tab 6: Word Cloud & Frequency
# ================================
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

with tab6:
    st.header("☁ Word Cloud & Frequency")
    st.write("Visualize the most frequent words in your text.")

    input_text = st.text_area("Enter text for word cloud:", height=150, placeholder="Paste text here...")
    if st.button("☁ Generate Word Cloud"):
        if input_text.strip():
            words = input_text.split()
            word_freq = Counter(words)

            # Display top 15 words
            freq_df = pd.DataFrame(word_freq.most_common(15), columns=["Word", "Frequency"])
            st.dataframe(freq_df, use_container_width=True)

            # Word Cloud generation
            wc = WordCloud(width=800, height=400, background_color="white").generate(input_text)

            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # Download word frequency CSV
            st.download_button(
                "⬇ Export Word Frequency",
                freq_df.to_csv(index=False).encode("utf-8"),
                "word_frequency.csv",
                "text/csv"
            )
        else:
            st.warning("Please enter some text first.")


# ================================
# Tab 7: Text Summarizer
# ================================
from transformers import pipeline

with tab7:
    st.header("📘 Text Summarizer")
    st.write("Summarize long text into concise, meaningful sentences using AI.")

    input_text = st.text_area("Enter text to summarize:", height=200, placeholder="Paste a long paragraph or article here...")
    if st.button("🧠 Summarize Text"):
        if input_text.strip():
            with st.spinner("Generating summary... please wait ⏳"):
                try:
                    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                    summary = summarizer(input_text, max_length=150, min_length=30, do_sample=False)
                    summarized_text = summary[0]["summary_text"]

                    st.subheader("📝 Summary:")
                    st.success(summarized_text)

                    # Download summary
                    st.download_button(
                        "⬇ Download Summary",
                        summarized_text.encode("utf-8"),
                        "summary.txt",
                        "text/plain"
                    )
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
        else:
            st.warning("Please enter text to summarize.")


# ===========================================
# 💅 Sidebar & Main Page Theming Fix
# ===========================================
st.markdown(f"""
    <style>
        /* Sidebar background & text */
        [data-testid="stSidebar"] {{
            background-color: {'#1e1e2f' if dark_mode else '#f9f9f9'};
            color: {'#f1f1f1' if dark_mode else '#111111'};
            transition: all 0.3s ease-in-out;
        }}

        /* Sidebar headers and section titles */
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h4 {{
            color: {'#f8f9fa' if dark_mode else '#111111'} !important;
        }}

        /* Sidebar labels */
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{
            color: {'#eaeaea' if dark_mode else '#333333'} !important;
        }}

        /* Sidebar selectboxes and checkboxes */
        div[data-baseweb="select"] > div {{
            background-color: {'#2b2b3d' if dark_mode else '#ffffff'} !important;
            color: {'#f8f9fa' if dark_mode else '#000000'} !important;
            border-radius: 8px;
            border: 1px solid {'#444' if dark_mode else '#ccc'} !important;
        }}

        div[role="radiogroup"] label p,
        div[role="checkbox"] label p {{
            color: {'#fafafa' if dark_mode else '#222222'} !important;
            font-size: 15px;
            font-weight: 500;
        }}

        /* Expander background */
        div[data-testid="stExpander"] {{
            background-color: {'#2b2b3d' if dark_mode else '#ffffff'} !important;
            border-radius: 10px;
            border: 1px solid {'#333' if dark_mode else '#ddd'} !important;
        }}

        /* Tabs text color fix */
        button[data-baseweb="tab"] {{
            color: {'#f8f9fa' if dark_mode else '#111111'} !important;
            font-weight: 600;
            font-size: 15px;
        }}
        button[data-baseweb="tab"]:hover {{
            background-color: {'#333c52' if dark_mode else '#e0e0e0'} !important;
        }}

        /* Main text area styling */
        textarea {{
            color: {'#ffffff' if dark_mode else '#000000'} !important; 
            background-color: {'#1e1e1e' if dark_mode else '#ffffff'} !important;
        }}

        /* Streamlit metric and dataframe color fix */
        div[data-testid="stDataFrame"] {{
            color: {'#e0e0e0' if dark_mode else '#000000'} !important;
        }}

    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
        /* Make all tab text white */
        button[data-baseweb="tab"] p {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* Highlight the active tab with a subtle underline */
        button[data-baseweb="tab"][aria-selected="true"] {
            border-bottom: 3px solid #4ea8de !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }

        /* Optional: change hover color */
        button[data-baseweb="tab"]:hover p {
            color: #90caf9 !important;
        }
    </style>
""", unsafe_allow_html=True)




# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p>🔬 <b>PolyLingua NLP Studio</b> — Explore, Analyze, and Understand Language with AI.</p>
        <p>Supports multilingual NLP tasks including POS tagging, grammar correction, sentiment analysis, entity recognition, and more.</p>
        <p>💡 Developed with passion by <b>Vaishnavi, Saniya, Madhumati, and Revati</b></p>
    </div>
""", unsafe_allow_html=True)