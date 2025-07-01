import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import re
from chain import build_chain

# Function to extract video ID from YouTube link
def extract_video_id(url):
    # Handles various YouTube URL formats
    regex = r"(?:v=|youtu\.be/|embed/|v/|shorts/)([\w-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

# Function to fetch transcript
def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Error fetching transcript: {e}"

# Define answer_question before it is used
def answer_question(question, transcript):
    chain = build_chain(transcript)
    return chain.invoke(question)

st.title("YouTube RAG Chatbot")

youtube_url = st.text_input("Paste a YouTube video link:")

if youtube_url:
    video_id = extract_video_id(youtube_url)
    if video_id:
        with st.spinner("Fetching transcript..."):
            transcript = fetch_transcript(video_id)
        if transcript.startswith("Error"):
            st.error(transcript)
        else:
            st.success("Transcript fetched! Now ask a question about the video.")
            question = st.text_input("Ask a question about the video transcript:")
            if question:
                with st.spinner("Thinking..."):
                    answer = answer_question(question, transcript)
                st.markdown("**Answer:**")
                st.write(answer)
    else:
        st.error("Invalid YouTube link. Please check and try again.")