import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

import time
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="VisionFlow Video Insight Analyzer",
    page_icon="🎥",
    layout="wide"
)

st.title("VisionFlow Video AI Analyzer 🎥✨")
st.header("Unleashing the Power of Gemini 2.0 for Video Insights")

@st.cache_resource
def create_video_agent():
    return Agent(
        name="VisionFlow AI Agent",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

video_agent = create_video_agent()

uploaded_video = st.file_uploader(
    "Upload a video file for analysis",
    type=['mp4', 'mov', 'avi'],
    help="Drop a video here to explore AI-driven insights."
)

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_video.read())
        temp_video_path = temp_file.name

    st.video(temp_video_path, format="video/mp4")

    query = st.text_area(
        "What insights do you need from the video?",
        placeholder="E.g., Summarize key moments, detect trends, or extract specific details.",
        help="Enter a detailed query to get accurate analysis."
    )

    if st.button("Generate Insights", key="analyze_video_button"):
        if not query:
            st.warning("Please enter a query before starting the analysis.")
        else:
            try:
                st.info("Processing video... Please wait.")
                processed_file = upload_file(temp_video_path)
                while processed_file.state.name == "PROCESSING":
                    time.sleep(1)
                    processed_file = get_file(processed_file.name)

                ai_prompt = f"Analyze the video and provide insights: {query}"
                ai_response = video_agent.run(ai_prompt, videos=[processed_file])

                st.subheader("Analysis Results")
                st.markdown(ai_response.content)

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                Path(temp_video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video to begin your analysis journey.")

st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 150px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
