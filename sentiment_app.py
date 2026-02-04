import streamlit as st
from transformers import pipeline
import time

# Set page config - MUST be first
st.set_page_config(
    page_title="SentiScope Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Inject CSS with different method
def local_css():
    st.markdown("""
    <style>
        /* Force background color */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
        
        [data-testid="stHeader"] {
            background: transparent !important;
        }
        
        /* Text area */
        [data-testid="stTextArea"] textarea {
            border-radius: 10px !important;
            border: 2px solid #667eea !important;
            font-size: 16px !important;
        }
        
        /* Button */
        [data-testid="stButton"] button {
            background: white !important;
            color: #667eea !important;
            border-radius: 8px !important;
            padding: 10px 30px !important;
            font-weight: bold !important;
            border: none !important;
        }
        
        [data-testid="stButton"] button:hover {
            background: #f0f0f0 !important;
        }
        
        /* Positive box */
        .positive-box {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin: 20px 0;
        }
        
        /* Negative box */
        .negative-box {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin: 20px 0;
        }
        
        .result-text {
            color: white;
            font-size: 36px;
            font-weight: bold;
            margin: 0;
        }
        
        .confidence-text {
            color: white;
            font-size: 20px;
            margin-top: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Title
st.markdown("<h1 style='text-align: center; color: white; font-size: 48px;'>SentiScope</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 20px; margin-bottom: 30px;'>AI-Powered Sentiment Analysis Tool</p>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_analyzer = load_model()

# Input
user_input = st.text_area(
    "Enter your text here:",
    height=150,
    placeholder="Type something...",
)

# Button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_button = st.button("Analyze Sentiment", use_container_width=True)

# Analysis
if analyze_button and user_input.strip():
    with st.spinner('Analyzing...'):
        time.sleep(0.3)
        result = sentiment_analyzer(user_input)[0]
        sentiment = result['label']
        confidence = result['score']
        
        if sentiment == "POSITIVE":
            st.markdown(f"""
                <div class="positive-box">
                    <p class="result-text">POSITIVE</p>
                    <p class="confidence-text">Confidence: {confidence:.1%}</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
                <div class="negative-box">
                    <p class="result-text">NEGATIVE</p>
                    <p class="confidence-text">Confidence: {confidence:.1%}</p>
                </div>
            """, unsafe_allow_html=True)

elif analyze_button:
    st.warning("Please enter some text!")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Built using Hugging Face Transformers</p>", unsafe_allow_html=True)